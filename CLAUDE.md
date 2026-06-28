# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI backend that generates personalized "what should I buy next?" product recommendations. It pulls a user's purchase history and a candidate catalog from Salesforce, builds a single prompt, and asks an LLM to pick 1–3 products with reasoning. The only business endpoint is `POST /api/insights/next-purchase`; `GET /` serves `index.html` (a self-contained test UI).

## Commands

```bash
pip install -r requirements.txt          # install deps (use venv/ if present)
uvicorn main:app --reload                 # run dev server on :8000
./venv/bin/uvicorn main:app --reload      # run via the checked-out venv

pytest                                    # run all tests (25)
pytest test_app.py::test_llm_falls_back_to_deepseek   # run a single test
```

Docker: `CMD` honors `$PORT` (defaults to 8000) for Railway-style deploys.

## Architecture

```
main.py                index.html ──HTTP──► POST /api/insights/next-purchase
  └─ generate_insight()                          │
        ├─ database.py  ──SOQL──► Salesforce Grocery_Product__c
        │     get_user_purchases_last_month()  +  get_candidate_products()
        ├─ _slim_for_prompt()  → projects records to prompt-relevant fields only
        └─ _call_llm()  ──► DeepSeek / NVIDIA / OpenRouter (round-robin fallback)
```

### Request flow (`generate_insight`)
1. **Response cache** — keyed by `(user_id, normalized_input)`; a hit skips both Salesforce and the LLM.
2. **SFDC fetch** — purchases + candidates, behind a per-`user_id` cache; even a new input avoids re-hitting Salesforce within the TTL. Candidates exclude already-purchased IDs.
3. Early returns (not cached) when there are no purchases or no candidates.
4. **LLM call** — sends a two-message chat: a static `system` message (`SYSTEM_PROMPT`: the full recommendation hierarchy — preference match → refill cycles → price drops → logical sanity — plus the HIGHLIGHTS vocabulary and strict JSON schema) and a per-request `user` message (current date + slimmed purchases/candidates + user input). The static system text is identical every call so it forms DeepSeek's cacheable prefix; keep the data out of it.
5. Parse with `_extract_json` (tolerant of prose/code fences) → validate with the `InsightResponse` Pydantic model → cache → return.

`user_id` is currently hardcoded to `"default_user"`, and the purchase SOQL is not actually filtered by user — the whole pipeline is single-user.

### LLM provider layer (`_call_llm`)
- `_call_llm` and every `_call_*` take a **`messages` list** (system + user), not a prompt string. Providers, all OpenAI-shaped: `_call_deepseek`, `_call_nvidia`, `_call_openrouter`. Each is enabled only if its API key is set; OpenRouter is always present and kept **last** as the shared-cap fallback. Each enforces `response_format: {"type":"json_object"}` (DeepSeek's JSON mode needs the word "json" in the system prompt, which it has; NVIDIA NIM's small llama-3.1-8b returns prose without it). `_call_llm` also validates that each provider's body is JSON-parseable and falls through to the next provider if not.
- **DeepSeek thinking mode**: `deepseek-v4-flash` is a hybrid model. When `DEEPSEEK_THINKING` (default on), `_call_deepseek` adds `"thinking": {"type":"enabled"}` so the model reasons step-by-step through the refill/price hierarchy — the chain-of-thought returns in `reasoning_content` (ignored); the JSON answer is in `content`. Thinking traces are slower and consume tokens, so DeepSeek gets its own roomier `DEEPSEEK_TIMEOUT` (60s) and `DEEPSEEK_MAX_TOKENS` (4096); thinking mode disables `temperature`/`top_p`. If DeepSeek is slow or errors, `_call_llm` falls through to NVIDIA → OpenRouter.
- Base order is DeepSeek → NVIDIA → OpenRouter. `LLM_PRIMARY` (default `deepseek` if `DEEPSEEK_API_KEY` is set, else `nvidia` → `openrouter`) is sorted to the front; `_call_llm` then tries **each provider once per round** for up to `LLM_MAX_ROUNDS` (3), pausing between rounds. This rotates across providers rather than exhausting one — every free tier fails transiently.
- OpenRouter uses its `models` array for in-request fallback (**max 3 entries**, so pick fallbacks from different provider pools than the primary). Retries transient failures with 429-aware backoff (longer cooldown on 429 since free limits are per-minute). See the [free-models memory] note — keep the 3-model free fallback array.
- A shared `requests.Session` is reused in both `main.py` and `database.py` to avoid repeated TLS handshakes.

### Salesforce layer (`database.py`)
- Client-credentials OAuth; token cached in module globals and auto-refreshed on a 401 retry.
- `_PRODUCT_FIELDS` limits the SOQL `SELECT` to columns consumed downstream — do not expand it without reason (it bloats both the SOQL payload and the prompt). `_PROMPT_FIELDS` in `main.py` is the analogous prompt-side projection.
- Records map `title__c` (fallback `Name`) onto `Products_Name__c` for prompt compatibility.

## Configuration (`.env`, gitignored)

Provider keys: `OPENROUTER_API_KEY` (required), `DEEPSEEK_API_KEY`, `NVIDIA_API_KEY` (each optional, enables that provider). Model overrides: `DEEPSEEK_MODEL`, `NVIDIA_MODEL`, `OPENROUTER_MODEL`, `OPENROUTER_FALLBACK_MODELS` (comma-separated). Tuning: `LLM_PRIMARY`, `OPENROUTER_TIMEOUT`; DeepSeek-specific: `DEEPSEEK_THINKING` (default `true`), `DEEPSEEK_TIMEOUT` (default `60`), `DEEPSEEK_MAX_TOKENS` (default `4096`). Salesforce: `SF_TOKEN_URL`, `SF_CLIENT_ID`, `SF_CLIENT_SECRET`.

## Testing notes

- `test_app.py` uses FastAPI `TestClient`. Two autouse fixtures matter: one **clears the in-memory caches** before/after each test, and another forces `LLM_PRIMARY=openrouter` with DeepSeek/NVIDIA keys nulled so real `.env` keys can never trigger live network calls. Re-enable a provider in a test by patching both its key and its `_call_*` function.
- Tests patch `main.get_user_purchases_last_month` / `get_candidate_products` / `_call_openrouter` etc. — Salesforce and the LLM are never hit for real.
