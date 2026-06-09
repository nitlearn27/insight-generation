import os
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from database import get_user_purchases_last_month, get_candidate_products

app = FastAPI(title="Insight Generation API")

# Add CORS middleware for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure LLM provider
load_dotenv() # Load variables from .env file

# Active provider: OpenRouter (OpenAI-compatible chat completions API).
# Gemini is parked for potential future use (see GEMINI_API_KEY in .env).
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Large reasoning models (e.g. gpt-oss-120b) can take ~20-60s+, so keep a generous
# timeout and retry transient failures (timeouts, 5xx, empty/blank completions).
OPENROUTER_TIMEOUT = 180
OPENROUTER_MAX_RETRIES = 3


def _call_openrouter(prompt: str) -> str:
    """Sends the prompt to OpenRouter and returns the raw message content (expected JSON).

    Retries on transient errors (network/timeouts, 5xx responses, empty completions)
    with a short backoff, raising the last error if all attempts fail.
    """
    last_err = None
    for attempt in range(OPENROUTER_MAX_RETRIES):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=OPENROUTER_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
            last_err = ValueError("OpenRouter returned an empty completion")
        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError) as e:
            last_err = e

        # Back off before the next attempt (skip the wait after the final try)
        if attempt < OPENROUTER_MAX_RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"OpenRouter request failed after {OPENROUTER_MAX_RETRIES} attempts: {last_err}"
    )


def _extract_json(text: str) -> str:
    """Defensively pulls the JSON object out of the model output (strips any prose/code fences)."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text

# --- In-memory caching (30 min TTL) ---
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

# Full response cache: key = (user_id, normalized user_input) -> (timestamp, InsightResponse)
_response_cache: dict = {}
# SFDC product data cache: key = user_id -> (timestamp, (recent_purchases, candidates))
_sfdc_cache: dict = {}


def _get_fresh(cache: dict, key):
    """Returns the cached value if present and within the TTL window, else None."""
    entry = cache.get(key)
    if entry and (time.time() - entry[0]) < CACHE_TTL_SECONDS:
        return entry[1]
    return None

# --- Pydantic Models ---

class InsightRequest(BaseModel):
    user_input: str

class Recommendation(BaseModel):
    product_name: str
    product_url: str
    price: float
    reasoning: str
    rating: str | None = None
    highlights: List[str] = []

class InsightResponse(BaseModel):
    insight_message: str
    recommendations: List[Recommendation]

# --- Endpoints ---

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.post("/api/insights/next-purchase", response_model=InsightResponse)
async def generate_insight(request: InsightRequest):
    """
    Generates a personalized product recommendation based on the user's
    recent purchase history and the current product catalog.
    """
    user_id = "default_user"
    cache_input = request.user_input.strip().lower()

    # 0. Response cache: identical input within the TTL skips both SFDC and Gemini
    cached_response = _get_fresh(_response_cache, (user_id, cache_input))
    if cached_response is not None:
        return cached_response

    # 1. Fetch user's recent purchases + candidates (SFDC), using a per-user cache
    #    so even a brand-new input avoids re-hitting Salesforce within the TTL.
    sfdc_data = _get_fresh(_sfdc_cache, user_id)
    if sfdc_data is not None:
        recent_purchases, candidates = sfdc_data
    else:
        recent_purchases = get_user_purchases_last_month(user_id)
        purchased_ids = [p["id"] for p in recent_purchases]
        # Fetch candidate products from catalog (exclude already purchased)
        candidates = get_candidate_products(purchased_ids, limit=20) if recent_purchases else []
        _sfdc_cache[user_id] = (time.time(), (recent_purchases, candidates))

    if not recent_purchases:
        # Fallback if no history (not cached — cheap and state-dependent)
        return InsightResponse(
            insight_message="You don't have any recent purchases. Explore our catalog!",
            recommendations=[]
        )

    if not candidates:
        return InsightResponse(
            insight_message="You've bought all our top products! Check back later.",
            recommendations=[]
        )

    # 2. Generate Insight via LLM
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is not set.")

    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    You are an AI shopping assistant. Your goal is to recommend 1 to 3 excellent products for the user to buy next, 
    based on their purchase history, their preferred store, and their custom preferences.

    Current Date: {current_date}

    User's recent purchases:
    {json.dumps(recent_purchases, indent=2)}

    Candidate products to choose from:
    {json.dumps(candidates, indent=2)}

    User Input Preference: "{request.user_input}"

    CRITICAL RECOMMENDATION HIERARCHY & LOGIC:
    1. PRIORITY 1: Match User Input Preference:
       - Carefully analyze the "User Input Preference" (e.g., store name like Amazon/Flipkart, specific item categories, brand names, or notes).
       - Select products from the candidate list that match this preference. If the user specifies a store preference like "Amazon" or "Flipkart" in their input, select ONLY products from that store.

    2. PRIORITY 2: Inventory Refill & Purchase Cycles (Compare "purchase_date" to the Current Date):
       - Perishable/High-Frequency Items (e.g., Milk, Curd, Dairy, Fresh Bread): These expire or get consumed quickly (cannot be stored for more than 2 days). If the user last purchased this type of item 2 or 3 days ago (or more), recommend a refill.
       - Weekly Vegetables & Staples (e.g., Potatoes, Tomatoes, Onions, Coriander): Typically last about 7 days. If the user last purchased this type of item 7 or more days ago, their stock is likely empty, and you should recommend a refill.
       - Regular Purchase Cycles: Analyze the user's purchase history. If they purchase a specific item regularly (e.g., every 4 to 5 days) and the time elapsed since their last purchase matches or exceeds that frequency, recommend a refill.

    3. PRIORITY 3: Price Drops:
       - Compare the candidate product's "current_price__c" with the user's "last_purchased_price__c" for that same product type. If the current price is lower than the price they paid last time, recommend it.
       - Highlight this price drop in your reasoning (e.g., "This item is currently on sale for ₹X, down from the ₹Y you paid on Z!").

    4. LOGICAL THINKING:
       - Think logically before recommending. For instance, do not recommend fresh tomatoes if they bought a large quantity yesterday, but do recommend it if they bought a small quantity 6 days ago.
       - Ensure the reasoning explicitly details these calculations (e.g., "Since you haven't purchased milk since June 1st and it has a 2-day freshness shelf life, we suggest refilling..." or "This tomato is now ₹26, which is cheaper than the ₹38 you paid on May 30th!").

    HIGHLIGHTS (why we recommend):
       - For each recommendation, include a "highlights" array of 1 to 3 short tags that summarize WHY it is recommended.
       - Choose from this controlled vocabulary wherever applicable, and keep them consistent with your reasoning and the hierarchy above:
         "Matches Preference" (Priority 1), "Refill Needed" (Priority 2), "Price Drop" (Priority 3), "Offer", "Top Rated", "Frequently Bought".
       - Only if none of the above fit, you may add a short custom tag (2-3 words).
       - Order the tags by relevance, most important first.

    Return your response strictly in the following JSON format:
    {{
      "insight_message": "A friendly introductory message summarizing the recommendations and highlighting the logical reasons (e.g., matching their input preferences, need for refills, or price drops).",
      "recommendations": [
        {{
          "product_name": "<the product name>",
          "product_url": "<the product url>",
          "price": <number>,
          "reasoning": "<1-2 sentence explanation of why they should buy this product based on the logic above>",
          "rating": "<rating string>",
          "highlights": ["<short tag>", "<short tag>"]
        }}
      ]
    }}
    """

    try:
        response_text = _call_openrouter(prompt)

        # Parse the JSON response (tolerant of surrounding prose/code fences)
        result_dict = json.loads(_extract_json(response_text))

        # Validate using Pydantic
        insight_response = InsightResponse(**result_dict)

        # Cache the successful LLM result keyed by normalized input
        _response_cache[(user_id, cache_input)] = (time.time(), insight_response)

        return insight_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insight: {str(e)}")

