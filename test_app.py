import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import main
from main import app, InsightResponse

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset the module-level in-memory caches before each test to avoid leakage."""
    main._response_cache.clear()
    main._sfdc_cache.clear()
    yield
    main._response_cache.clear()
    main._sfdc_cache.clear()


# Test 1: Successful generation of insight using mock Gemini client
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_generate_insight_success(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    # Mock data
    mock_get_purchases.return_value = [
        {
            "id": "prod_2",
            "Products_Name__c": "Apple MacBook Pro 14-inch (M3 Pro)",
            "last_purchased_price__c": 1999.00,
            "purchase_date": "2026-05-15T10:00:00Z"
        }
    ]
    mock_get_candidates.return_value = [
        {
            "id": "prod_6",
            "Products_Name__c": "Twelve South Curve Laptop Stand",
            "model__c": "TS-2101",
            "original_price__c": 59.99,
            "rating__c": "4.8",
            "review_count__c": 4500,
            "number_of_times_purchased__c": 8000,
            "product_url__c": "https://example.com/products/twelve-south-curve",
            "specifications__c": "Matte white/black, ergonomic design",
        }
    ]

    # Mock response from Gemini model
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "insight_message": "Based on your recent premium purchases, we have a top recommendation to enhance your new workstation setup.",
        "recommendations": [
            {
                "product_name": "Twelve South Curve Laptop Stand",
                "product_url": "https://example.com/products/twelve-south-curve",
                "price": 59.99,
                "reasoning": "You've recently invested in a high-end MacBook Pro and a large external monitor. The Twelve South Curve Laptop Stand will perfectly complement your new setup by providing ergonomic positioning for your MacBook, improving posture and desk organization while using your external display.",
                "rating": "4.8"
            }
        ]
    })
    
    mock_gemini_client.models.generate_content.return_value = mock_response

    # Temporarily ensure gemini_client is not None (in case environment variable wasn't set)
    with patch("main.gemini_client", mock_gemini_client):
        response = client.post(
            "/api/insights/next-purchase",
            json={"user_input": "Give me a recommendation"}
        )

    assert response.status_code == 200
    data = response.json()
    assert "insight_message" in data
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["product_name"] == "Twelve South Curve Laptop Stand"
    assert data["recommendations"][0]["price"] == 59.99


# Test 2: Fallback when user has no recent purchase history
@patch("main.get_user_purchases_last_month")
@patch("main.gemini_client")
def test_generate_insight_no_history(mock_gemini_client, mock_get_purchases):
    mock_get_purchases.return_value = []

    response = client.post(
        "/api/insights/next-purchase",
        json={"user_input": "Suggest something"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["insight_message"] == "You don't have any recent purchases. Explore our catalog!"
    assert data["recommendations"] == []
    # Assert Gemini API was never called
    mock_gemini_client.models.generate_content.assert_not_called()


# Test 3: Fallback when candidate list is empty
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_generate_insight_no_candidates(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Headphones"}]
    mock_get_candidates.return_value = []

    response = client.post(
        "/api/insights/next-purchase",
        json={"user_input": "Find items"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["insight_message"] == "You've bought all our top products! Check back later."
    assert data["recommendations"] == []
    # Assert Gemini API was never called
    mock_gemini_client.models.generate_content.assert_not_called()


# Test 4: Missing GEMINI_API_KEY config
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
def test_generate_insight_missing_api_key(mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Headphones"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Keyboard"}]

    with patch("main.gemini_client", None):
        response = client.post(
            "/api/insights/next-purchase",
            json={"user_input": "Give recommendations"}
        )

    assert response.status_code == 500
    assert "GEMINI_API_KEY environment variable is not set." in response.json()["detail"]


# Test 5: Gemini API errors out
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_generate_insight_gemini_error(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Headphones"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Keyboard"}]

    mock_gemini_client.models.generate_content.side_effect = Exception("API rate limit exceeded")

    with patch("main.gemini_client", mock_gemini_client):
        response = client.post(
            "/api/insights/next-purchase",
            json={"user_input": "What to buy?"}
        )

    assert response.status_code == 500
    assert "Failed to generate insight: API rate limit exceeded" in response.json()["detail"]


# Test 6: Request validation error (e.g. missing user_input)
def test_generate_insight_validation_error():
    response = client.post(
        "/api/insights/next-purchase",
        json={}  # Missing user_input
    )
    assert response.status_code == 422
    assert "detail" in response.json()


# Test 7: User Input parsing and multiple recommendations success
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_generate_insight_preferences_and_multiple_recs(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    # Mock data
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Tata Salt"}]
    mock_get_candidates.return_value = [
        {"id": "prod_2", "title__c": "Fresh Tomato", "Products_Name__c": "Fresh Tomato", "source__c": "Amazon", "current_price__c": 26.0, "rating__c": "4.5", "product_url__c": "https://amazon.com/..."},
        {"id": "prod_3", "title__c": "Apple Royal Gala", "Products_Name__c": "Apple Royal Gala", "source__c": "Amazon", "current_price__c": 91.0, "rating__c": "4.6", "product_url__c": "https://amazon.com/..."}
    ]

    # Mock response from Gemini model returning multiple recommendations
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "insight_message": "Based on your interest in fresh ingredients and organic groceries, here are top recommendations from Amazon.",
        "recommendations": [
            {
                "product_name": "Fresh Tomato",
                "product_url": "https://amazon.com/...",
                "price": 26.0,
                "reasoning": "A must-have for daily cooking, matching your recent purchases.",
                "rating": "4.5"
            },
            {
                "product_name": "Apple Royal Gala",
                "product_url": "https://amazon.com/...",
                "price": 91.0,
                "reasoning": "A healthy fruit preference to add to your daily diet.",
                "rating": "4.6"
            }
        ]
    })
    mock_gemini_client.models.generate_content.return_value = mock_response

    with patch("main.gemini_client", mock_gemini_client):
        response = client.post(
            "/api/insights/next-purchase",
            json={
                "user_input": "I only want organic items from Amazon"
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert "insight_message" in data
    assert len(data["recommendations"]) == 2
    assert data["recommendations"][0]["product_name"] == "Fresh Tomato"
    assert data["recommendations"][1]["product_name"] == "Apple Royal Gala"
    
    # Assert database query was called with the candidates limit = 20
    mock_get_candidates.assert_called_once_with(["prod_1"], limit=20)


# Test 8: Response validation matches all model fields
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_response_fields_completeness(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Tata Salt"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "title__c": "Fresh Tomato", "Products_Name__c": "Fresh Tomato", "source__c": "Amazon"}]

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "insight_message": "Insight summary text.",
        "recommendations": [
            {
                "product_name": "Test Product",
                "product_url": "https://example.com",
                "price": 10.5,
                "reasoning": "Reasoning details.",
                "rating": "4.5"
            }
        ]
    })
    mock_gemini_client.models.generate_content.return_value = mock_response

    with patch("main.gemini_client", mock_gemini_client):
        response = client.post(
            "/api/insights/next-purchase",
            json={"user_input": "Test completeness"}
        )

    assert response.status_code == 200
    data = response.json()
    
    # Assert top-level fields are present
    assert "insight_message" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 1
    
    # Assert all Recommendation subfields are present and correctly typed
    rec = data["recommendations"][0]
    assert "product_name" in rec and isinstance(rec["product_name"], str)
    assert "product_url" in rec and isinstance(rec["product_url"], str)
    assert "price" in rec and isinstance(rec["price"], (int, float))
    assert "reasoning" in rec and isinstance(rec["reasoning"], str)
    assert "rating" in rec and isinstance(rec["rating"], str)


# Test 9: Highlights array round-trips in the response
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_generate_insight_highlights(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk", "source__c": "Amazon"}]

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "insight_message": "Time to refill your milk.",
        "recommendations": [
            {
                "product_name": "Fresh Milk",
                "product_url": "https://amazon.com/milk",
                "price": 30.0,
                "reasoning": "You last bought milk 3 days ago and it's now cheaper.",
                "rating": "4.5",
                "highlights": ["Refill Needed", "Price Drop"]
            }
        ]
    })
    mock_gemini_client.models.generate_content.return_value = mock_response

    with patch("main.gemini_client", mock_gemini_client):
        response = client.post("/api/insights/next-purchase", json={"user_input": "milk"})

    assert response.status_code == 200
    rec = response.json()["recommendations"][0]
    assert "highlights" in rec
    assert isinstance(rec["highlights"], list)
    assert all(isinstance(h, str) for h in rec["highlights"])
    assert rec["highlights"] == ["Refill Needed", "Price Drop"]


# Test 10: highlights defaults to [] when the LLM omits it
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_generate_insight_highlights_default(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk"}]

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "insight_message": "A recommendation.",
        "recommendations": [
            {
                "product_name": "Fresh Milk",
                "product_url": "https://amazon.com/milk",
                "price": 30.0,
                "reasoning": "Good pick.",
                "rating": "4.5"
            }
        ]
    })
    mock_gemini_client.models.generate_content.return_value = mock_response

    with patch("main.gemini_client", mock_gemini_client):
        response = client.post("/api/insights/next-purchase", json={"user_input": "milk"})

    assert response.status_code == 200
    assert response.json()["recommendations"][0]["highlights"] == []


# Test 11: Response cache — identical input within TTL skips both SFDC and Gemini
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_response_cache_hit_same_input(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk"}]

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "insight_message": "Cached message.",
        "recommendations": [
            {
                "product_name": "Fresh Milk",
                "product_url": "https://amazon.com/milk",
                "price": 30.0,
                "reasoning": "Good pick.",
                "rating": "4.5",
                "highlights": ["Top Rated"]
            }
        ]
    })
    mock_gemini_client.models.generate_content.return_value = mock_response

    with patch("main.gemini_client", mock_gemini_client):
        first = client.post("/api/insights/next-purchase", json={"user_input": "milk please"})
        # Same input, different casing/whitespace -> normalized to the same cache key
        second = client.post("/api/insights/next-purchase", json={"user_input": "  Milk Please  "})

    assert first.status_code == 200 and second.status_code == 200
    assert first.json() == second.json()
    # Gemini and SFDC should each only run once (second request served from cache)
    assert mock_gemini_client.models.generate_content.call_count == 1
    assert mock_get_purchases.call_count == 1
    assert mock_get_candidates.call_count == 1


# Test 12: SFDC cache — different input reruns Gemini but reuses SFDC data
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main.gemini_client")
def test_sfdc_cache_hit_different_input(mock_gemini_client, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk"}]

    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "insight_message": "A recommendation.",
        "recommendations": [
            {
                "product_name": "Fresh Milk",
                "product_url": "https://amazon.com/milk",
                "price": 30.0,
                "reasoning": "Good pick.",
                "rating": "4.5",
                "highlights": ["Top Rated"]
            }
        ]
    })
    mock_gemini_client.models.generate_content.return_value = mock_response

    with patch("main.gemini_client", mock_gemini_client):
        client.post("/api/insights/next-purchase", json={"user_input": "milk"})
        client.post("/api/insights/next-purchase", json={"user_input": "bread"})

    # Distinct inputs -> two Gemini calls, but SFDC fetched only once (cache hit)
    assert mock_gemini_client.models.generate_content.call_count == 2
    assert mock_get_purchases.call_count == 1
    assert mock_get_candidates.call_count == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
