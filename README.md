# 💹 Agentic AI Financial Advisor

This project is my end-to-end attempt at building a fully local, research-backed **agentic financial advisor**. The system profiles the user, researches companies using multiple specialized agents, retrieves real SEC filings, and then produces a diversified investment strategy — all through an AI agent workflow instead of a single model.

The goal was to show what a real, research-driven AI advisor could look like when each step is handled by an agent with a clear role and toolset.

---

## 🚀 Overview

### What the system does:
- Builds a **structured investor profile** through a conversational onboarding agent  
- Selects a custom ticker universe based on the user's interests  
- Runs a full multi-agent **research pipeline**:
  - Market / quantitative analysis  
  - News + sentiment analysis (Finnhub + local LLM summarization)
  - SEC filings research (RAG over online 10-Ks + FAISS fallback)
- Generates a **personalized portfolio allocation** with rationale for each position  
- Displays everything inside a clean Streamlit UI

---

## 🧱 Architecture

### 🔹 1. Profiling Agent (`profiling_crew.py`)
- Conversational onboarding
- Collects risk tolerance, investment horizon, investment amount, sectors, notes, watchlist, etc.
- Produces a structured `UserProfile` object

### 🔹 2. Ticker Planner (`ticker_planner.py`)
- LLM-only JSON output
- Builds a 4-part universe:
  - primary_tickers  
  - similar_tickers  
  - balancing_tickers  
  - other_tickers  

### 🔹 3. Research Crew (`investment_strategy_crew.py`)
Runs **four agents in sequence**:

| Agent | Job |
|-------|-----|
| Market & Quant | Uses yfinance + custom metrics |
| News & Sentiment | Calls Finnhub + local LLM summarizer |
| SEC Question Designer | Turns market/news signals into SEC research questions |
| SEC RAG Agent | Runs retrieval over real filings |

Outputs research in a per-ticker structured format.

### 🔹 4. Portfolio Decision Agent (`portfolio_decision_crew.py`)
- Uses OpenAI’s `gpt-4o-mini`  
- Converts research into a diversified portfolio  
- Assigns buckets (core / growth / speculative)  
- Produces final JSON with weights + rationale

---

## 📊 Streamlit UI (`app.py`)

The UI includes:

- Chat-based onboarding  
- Profile snapshot panel  
- “Generate Investment Strategy” button  
- Tabs for:
  - **Strategy & Allocation** (table + bar chart)
  - **Per-Ticker Research** (market, sentiment, SEC)
  - Raw developer logs

---

## 📁 Project Structure
