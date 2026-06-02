# Insight Generation API

This repository contains the backend service for generating personalized "What should I buy next?" insights based on a user's purchase history and a product catalog.

## Tech Stack
- Python 3.x
- FastAPI
- Uvicorn
- LLM Integration (e.g., Google Generative AI or OpenAI)

## Features
- Provides a REST API endpoint (`POST /api/insights/next-purchase`) to generate AI-driven product recommendations.
- Analyzes recent purchase history and catalog items to find the best match.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```
