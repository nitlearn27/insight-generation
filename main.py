import os
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
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

# Configure Gemini API Client
load_dotenv() # Load variables from .env file

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# --- Pydantic Models ---

class InsightRequest(BaseModel):
    user_input: str

class Recommendation(BaseModel):
    product_name: str
    product_url: str
    price: float
    reasoning: str
    rating: str | None = None

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
    # 1. Fetch user's recent purchases
    recent_purchases = get_user_purchases_last_month("default_user")
    if not recent_purchases:
        # Fallback if no history
        return InsightResponse(
            insight_message="You don't have any recent purchases. Explore our catalog!",
            recommendations=[]
        )

    # 2. Extract purchased IDs
    purchased_ids = [p["id"] for p in recent_purchases]

    # 3. Fetch candidate products from catalog (exclude already purchased)
    candidates = get_candidate_products(purchased_ids, limit=20)
    
    if not candidates:
        return InsightResponse(
            insight_message="You've bought all our top products! Check back later.",
            recommendations=[]
        )

    # 4. Generate Insight via LLM
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set.")

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

    Return your response strictly in the following JSON format:
    {{
      "insight_message": "A friendly introductory message summarizing the recommendations and highlighting the logical reasons (e.g., matching their input preferences, need for refills, or price drops).",
      "recommendations": [
        {{
          "product_name": "<the product name>",
          "product_url": "<the product url>",
          "price": <number>,
          "reasoning": "<1-2 sentence explanation of why they should buy this product based on the logic above>",
          "rating": "<rating string>"
        }}
      ]
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        # Parse the JSON response
        result_dict = json.loads(response.text)
        
        # Validate and return using Pydantic
        return InsightResponse(**result_dict)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insight: {str(e)}")

