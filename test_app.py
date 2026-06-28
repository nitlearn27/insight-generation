import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import main
from main import app

client = TestClient(app)

# Providers now take a messages list (system + user) rather than a prompt string.
MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset the module-level in-memory caches before each test to avoid leakage."""
    main._response_cache.clear()
    main._sfdc_cache.clear()
    yield
    main._response_cache.clear()
    main._sfdc_cache.clear()


@pytest.fixture(autouse=True)
def openrouter_primary_no_fallbacks(monkeypatch):
    """Default tests to OpenRouter-primary with the fallback providers disabled
    so real keys in .env can never cause live network calls. Individual tests
    re-enable a provider by patching its key and _call_* function."""
    monkeypatch.setattr(main, "LLM_PRIMARY", "openrouter")
    monkeypatch.setattr(main, "DEEPSEEK_API_KEY", None)
    monkeypatch.setattr(main, "NVIDIA_API_KEY", None)


# Test 1: Successful generation of insight using mock OpenRouter call
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main._call_openrouter")
def test_generate_insight_success(mock_call, mock_get_candidates, mock_get_purchases):
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

    # Mock response content from the LLM
    mock_call.return_value = json.dumps({
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
@patch("main._call_openrouter")
def test_generate_insight_no_history(mock_call, mock_get_purchases):
    mock_get_purchases.return_value = []

    response = client.post(
        "/api/insights/next-purchase",
        json={"user_input": "Suggest something"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["insight_message"] == "You don't have any recent purchases. Explore our catalog!"
    assert data["recommendations"] == []
    # Assert the LLM was never called
    mock_call.assert_not_called()


# Test 3: Fallback when candidate list is empty
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main._call_openrouter")
def test_generate_insight_no_candidates(mock_call, mock_get_candidates, mock_get_purchases):
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
    # Assert the LLM was never called
    mock_call.assert_not_called()


# Test 4: Missing OPENROUTER_API_KEY config
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
def test_generate_insight_missing_api_key(mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Headphones"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Keyboard"}]

    with patch("main.OPENROUTER_API_KEY", None):
        response = client.post(
            "/api/insights/next-purchase",
            json={"user_input": "Give recommendations"}
        )

    assert response.status_code == 500
    assert "OPENROUTER_API_KEY environment variable is not set." in response.json()["detail"]


# Test 5: LLM call errors out
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main._call_openrouter")
def test_generate_insight_llm_error(mock_call, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Headphones"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Keyboard"}]

    mock_call.side_effect = Exception("API rate limit exceeded")

    # The autouse fixture already disables DeepSeek/NVIDIA, so the OpenRouter
    # failure surfaces as a 500 (and the test never makes a real network call).
    with patch("main.time.sleep"):
        response = client.post(
            "/api/insights/next-purchase",
            json={"user_input": "What to buy?"}
        )

    assert response.status_code == 500
    assert "API rate limit exceeded" in response.json()["detail"]


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
@patch("main._call_openrouter")
def test_generate_insight_preferences_and_multiple_recs(mock_call, mock_get_candidates, mock_get_purchases):
    # Mock data
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Tata Salt"}]
    mock_get_candidates.return_value = [
        {"id": "prod_2", "title__c": "Fresh Tomato", "Products_Name__c": "Fresh Tomato", "source__c": "Amazon", "current_price__c": 26.0, "rating__c": "4.5", "product_url__c": "https://amazon.com/..."},
        {"id": "prod_3", "title__c": "Apple Royal Gala", "Products_Name__c": "Apple Royal Gala", "source__c": "Amazon", "current_price__c": 91.0, "rating__c": "4.6", "product_url__c": "https://amazon.com/..."}
    ]

    # Mock LLM content returning multiple recommendations
    mock_call.return_value = json.dumps({
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
@patch("main._call_openrouter")
def test_response_fields_completeness(mock_call, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Tata Salt"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "title__c": "Fresh Tomato", "Products_Name__c": "Fresh Tomato", "source__c": "Amazon"}]

    mock_call.return_value = json.dumps({
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
@patch("main._call_openrouter")
def test_generate_insight_highlights(mock_call, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk", "source__c": "Amazon"}]

    mock_call.return_value = json.dumps({
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
@patch("main._call_openrouter")
def test_generate_insight_highlights_default(mock_call, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk"}]

    mock_call.return_value = json.dumps({
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

    response = client.post("/api/insights/next-purchase", json={"user_input": "milk"})

    assert response.status_code == 200
    assert response.json()["recommendations"][0]["highlights"] == []


# Test 11: Response cache — identical input within TTL skips both SFDC and the LLM
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main._call_openrouter")
def test_response_cache_hit_same_input(mock_call, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk"}]

    mock_call.return_value = json.dumps({
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

    first = client.post("/api/insights/next-purchase", json={"user_input": "milk please"})
    # Same input, different casing/whitespace -> normalized to the same cache key
    second = client.post("/api/insights/next-purchase", json={"user_input": "  Milk Please  "})

    assert first.status_code == 200 and second.status_code == 200
    assert first.json() == second.json()
    # LLM and SFDC should each only run once (second request served from cache)
    assert mock_call.call_count == 1
    assert mock_get_purchases.call_count == 1
    assert mock_get_candidates.call_count == 1


# Test 12: SFDC cache — different input reruns the LLM but reuses SFDC data
@patch("main.get_user_purchases_last_month")
@patch("main.get_candidate_products")
@patch("main._call_openrouter")
def test_sfdc_cache_hit_different_input(mock_call, mock_get_candidates, mock_get_purchases):
    mock_get_purchases.return_value = [{"id": "prod_1", "Products_Name__c": "Milk"}]
    mock_get_candidates.return_value = [{"id": "prod_2", "Products_Name__c": "Fresh Milk"}]

    mock_call.return_value = json.dumps({
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

    client.post("/api/insights/next-purchase", json={"user_input": "milk"})
    client.post("/api/insights/next-purchase", json={"user_input": "bread"})

    # Distinct inputs -> two LLM calls, but SFDC fetched only once (cache hit)
    assert mock_call.call_count == 2
    assert mock_get_purchases.call_count == 1
    assert mock_get_candidates.call_count == 1


# Test 13: _call_openrouter retries a transient timeout, then succeeds
def test_call_openrouter_retries_then_succeeds():
    good = MagicMock()
    good.raise_for_status.return_value = None
    good.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}

    side_effects = [main.requests.exceptions.Timeout("slow"), good]
    with patch("main.time.sleep"), \
         patch("main._session.post", side_effect=side_effects) as mock_post:
        result = main._call_openrouter(MESSAGES)

    assert result == '{"ok": true}'
    assert mock_post.call_count == 2


# Test 14: _call_openrouter raises after exhausting all retries
def test_call_openrouter_raises_after_retries():
    with patch("main.time.sleep"), \
         patch("main._session.post", side_effect=main.requests.exceptions.Timeout("slow")) as mock_post:
        with pytest.raises(RuntimeError):
            main._call_openrouter(MESSAGES)

    assert mock_post.call_count == main.OPENROUTER_MAX_RETRIES


# Test 15: _call_openrouter retries when the model returns an empty completion
def test_call_openrouter_retries_on_empty_content():
    empty = MagicMock()
    empty.raise_for_status.return_value = None
    empty.json.return_value = {"choices": [{"message": {"content": "   "}}]}
    good = MagicMock()
    good.raise_for_status.return_value = None
    good.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}

    with patch("main.time.sleep"), \
         patch("main._session.post", side_effect=[empty, good]) as mock_post:
        result = main._call_openrouter(MESSAGES)

    assert result == '{"ok": true}'
    assert mock_post.call_count == 2


# Test 15b: _call_deepseek enables thinking mode and sends the messages + JSON mode
def test_deepseek_enables_thinking_mode():
    good = MagicMock()
    good.raise_for_status.return_value = None
    good.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}

    with patch("main.DEEPSEEK_THINKING", True), \
         patch("main._session.post", return_value=good) as mock_post:
        result = main._call_deepseek(MESSAGES)

    assert result == '{"ok": true}'
    body = mock_post.call_args.kwargs["json"]
    assert body["thinking"] == {"type": "enabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == MESSAGES


# Test 15c: _call_deepseek omits the thinking field when thinking mode is disabled
def test_deepseek_thinking_disabled():
    good = MagicMock()
    good.raise_for_status.return_value = None
    good.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}

    with patch("main.DEEPSEEK_THINKING", False), \
         patch("main._session.post", return_value=good) as mock_post:
        main._call_deepseek(MESSAGES)

    assert "thinking" not in mock_post.call_args.kwargs["json"]


# Test 16: _call_llm falls back to DeepSeek when the first OpenRouter attempt fails
def test_llm_falls_back_to_deepseek():
    with patch("main._call_openrouter", side_effect=RuntimeError("congested")) as mock_or, \
         patch("main.DEEPSEEK_API_KEY", "test-key"), \
         patch("main._call_deepseek", return_value='{"ok": 1}') as mock_deepseek:
        result = main._call_llm(MESSAGES)

    assert result == '{"ok": 1}'
    mock_or.assert_called_once_with(MESSAGES, max_retries=1)
    mock_deepseek.assert_called_once_with(MESSAGES)


# Test 17: _call_llm alternates back to OpenRouter when DeepSeek also fails
def test_llm_deepseek_fails_then_openrouter_retries():
    or_results = [RuntimeError("congested"), '{"ok": 2}']
    with patch("main._call_openrouter", side_effect=or_results) as mock_or, \
         patch("main.DEEPSEEK_API_KEY", "test-key"), \
         patch("main._call_deepseek", side_effect=RuntimeError("overloaded")) as mock_deepseek, \
         patch("main.time.sleep"):
        result = main._call_llm(MESSAGES)

    assert result == '{"ok": 2}'
    assert mock_or.call_count == 2
    mock_deepseek.assert_called_once()


# Test 17b: a provider that returns non-JSON prose falls through to the next provider
def test_llm_non_json_falls_through():
    with patch("main.NVIDIA_API_KEY", "test-key"), \
         patch("main.LLM_PRIMARY", "nvidia"), \
         patch("main._call_nvidia", return_value="Sure! Here are some picks for you.") as mock_nv, \
         patch("main._call_openrouter", return_value='{"ok": 9}') as mock_or, \
         patch("main.time.sleep"):
        result = main._call_llm(MESSAGES)

    assert result == '{"ok": 9}'
    mock_nv.assert_called()
    mock_or.assert_called()


# Test 18: _call_llm skips DeepSeek entirely when no DEEPSEEK_API_KEY is configured
def test_llm_skips_deepseek_without_key():
    or_results = [RuntimeError("congested"), '{"ok": 3}']
    with patch("main._call_openrouter", side_effect=or_results) as mock_or, \
         patch("main.DEEPSEEK_API_KEY", None), \
         patch("main._call_deepseek") as mock_deepseek, \
         patch("main.time.sleep"):
        result = main._call_llm(MESSAGES)

    assert result == '{"ok": 3}'
    mock_deepseek.assert_not_called()


# Test 19: with DeepSeek as primary, OpenRouter is never called when DeepSeek succeeds
def test_llm_deepseek_primary_called_first():
    with patch("main._call_openrouter") as mock_or, \
         patch("main.DEEPSEEK_API_KEY", "test-key"), \
         patch("main.LLM_PRIMARY", "deepseek"), \
         patch("main._call_deepseek", return_value='{"ok": 4}') as mock_deepseek:
        result = main._call_llm(MESSAGES)

    assert result == '{"ok": 4}'
    mock_deepseek.assert_called_once_with(MESSAGES)
    mock_or.assert_not_called()


# Test 20: full chain order — DeepSeek (primary) → NVIDIA → OpenRouter
def test_llm_full_chain_order():
    call_order = []
    with patch("main.DEEPSEEK_API_KEY", "test-key"), \
         patch("main.NVIDIA_API_KEY", "test-key"), \
         patch("main.LLM_PRIMARY", "deepseek"), \
         patch("main._call_deepseek", side_effect=lambda p: call_order.append("deepseek") or (_ for _ in ()).throw(RuntimeError("ds down"))), \
         patch("main._call_nvidia", side_effect=lambda p: call_order.append("nvidia") or (_ for _ in ()).throw(RuntimeError("nv down"))), \
         patch("main._call_openrouter", side_effect=lambda p, **kw: call_order.append("openrouter") or '{"ok": 5}'), \
         patch("main.time.sleep"):
        result = main._call_llm(MESSAGES)

    assert result == '{"ok": 5}'
    assert call_order[:3] == ["deepseek", "nvidia", "openrouter"]


# Test 22b: LLM_PRIMARY="nvidia" puts NVIDIA first; OpenRouter is never reached
def test_llm_nvidia_primary_called_first():
    with patch("main._call_openrouter") as mock_or, \
         patch("main.NVIDIA_API_KEY", "test-key"), \
         patch("main.LLM_PRIMARY", "nvidia"), \
         patch("main._call_nvidia", return_value='{"ok": 7}') as mock_nv:
        result = main._call_llm(MESSAGES)

    assert result == '{"ok": 7}'
    mock_nv.assert_called_once_with(MESSAGES)
    mock_or.assert_not_called()


# Test 22: _call_llm raises after all rounds of both providers fail
def test_llm_raises_after_all_rounds():
    with patch("main._call_openrouter", side_effect=RuntimeError("congested")) as mock_or, \
         patch("main.DEEPSEEK_API_KEY", "test-key"), \
         patch("main._call_deepseek", side_effect=RuntimeError("overloaded")) as mock_deepseek, \
         patch("main.time.sleep"):
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            main._call_llm(MESSAGES)

    assert mock_or.call_count == main.LLM_MAX_ROUNDS
    assert mock_deepseek.call_count == main.LLM_MAX_ROUNDS


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
