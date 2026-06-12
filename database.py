import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SF_TOKEN_URL = os.environ.get("SF_TOKEN_URL")
SF_CLIENT_ID = os.environ.get("SF_CLIENT_ID")
SF_CLIENT_SECRET = os.environ.get("SF_CLIENT_SECRET")

# Reuse one HTTP connection (TLS handshake) across Salesforce calls
_session = requests.Session()

# In-memory cache for the Salesforce access token
_cached_token = None
_instance_url = None

# Only the columns consumed downstream (prompt + response); fetching all 18 fields
# bloats both the SOQL payload and the LLM prompt.
_PRODUCT_FIELDS = (
    "Id, Name, title__c, brand__c, category__c, current_price__c, "
    "last_purchased_price__c, rating__c, product_url__c, source__c, "
    "last_ordered_date__c, number_of_times_purchased__c, availability__c"
)

def get_salesforce_token() -> tuple[str, str]:
    """Retrieves an OAuth2 access token from Salesforce using Client Credentials flow."""
    global _cached_token, _instance_url
    if _cached_token and _instance_url:
        return _cached_token, _instance_url

    if not SF_TOKEN_URL or not SF_CLIENT_ID or not SF_CLIENT_SECRET:
        raise ValueError("Salesforce environment variables (SF_TOKEN_URL, SF_CLIENT_ID, SF_CLIENT_SECRET) are not set.")

    payload = {
        "grant_type": "client_credentials",
        "client_id": SF_CLIENT_ID,
        "client_secret": SF_CLIENT_SECRET
    }
    
    response = _session.post(SF_TOKEN_URL, data=payload)
    response.raise_for_status()
    data = response.json()
    
    _cached_token = data["access_token"]
    _instance_url = data["instance_url"]
    return _cached_token, _instance_url

def execute_soql(query: str) -> List[Dict[str, Any]]:
    """Executes a SOQL query against the Salesforce REST API, with automatic 401 token refresh."""
    global _cached_token, _instance_url
    
    token, instance_url = get_salesforce_token()
    url = f"{instance_url}/services/data/v60.0/query/"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = _session.get(url, headers=headers, params={"q": query})

    # Handle token expiration (401 Unauthorized) by retrying once
    if response.status_code == 401:
        _cached_token = None
        _instance_url = None
        token, instance_url = get_salesforce_token()
        headers["Authorization"] = f"Bearer {token}"
        response = _session.get(url, headers=headers, params={"q": query})
        
    response.raise_for_status()
    return response.json().get("records", [])

def get_user_purchases_last_month(user_id: str) -> List[Dict[str, Any]]:
    """
    Fetches the user's recent purchases from the Salesforce Grocery_Product__c object.
    Filters for products that have a non-null last ordered date, sorted by newest.
    """
    soql = (
        f"SELECT {_PRODUCT_FIELDS} "
        "FROM Grocery_Product__c "
        "WHERE last_ordered_date__c != null "
        "ORDER BY last_ordered_date__c DESC "
        "LIMIT 10"
    )
    records = execute_soql(soql)

    recent_purchases = []
    for r in records:
        purchase_item = {
            "id": r["Id"],
            "purchase_date": r.get("last_ordered_date__c"),
        }
        # Copy all retrieved Salesforce fields dynamically
        for k, v in r.items():
            if k != 'attributes':
                purchase_item[k] = v
        # Map search compatibility keys
        purchase_item["Products_Name__c"] = r.get("title__c") or r.get("Name")

        recent_purchases.append(purchase_item)
    return recent_purchases

def get_candidate_products(purchased_ids: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetches the candidate products from the Salesforce Grocery_Product__c object.
    Excludes products that the user has already purchased recently.
    """
    # Filter, sort, and limit server-side instead of pulling the whole object
    # and post-processing in Python.
    soql = f"SELECT {_PRODUCT_FIELDS} FROM Grocery_Product__c"
    if purchased_ids:
        quoted_ids = ", ".join(f"'{pid}'" for pid in purchased_ids)
        soql += f" WHERE Id NOT IN ({quoted_ids})"
    soql += f" ORDER BY rating__c DESC NULLS LAST LIMIT {int(limit)}"
    records = execute_soql(soql)

    candidates = []
    for r in records:
        candidate_item = {
            "id": r["Id"]
        }
        # Copy all retrieved Salesforce fields dynamically
        for k, v in r.items():
            if k != 'attributes':
                candidate_item[k] = v
        # Map search compatibility keys
        candidate_item["Products_Name__c"] = r.get("title__c") or r.get("Name")

        candidates.append(candidate_item)

    return candidates
