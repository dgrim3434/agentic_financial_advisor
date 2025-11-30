**Agentic AI Financial Advisor**

This project is my implementation of a full end-to-end, agentic financial research system that behaves much closer to how an actual investment professional works — not just a black-box LLM spitting out answers. The goal was to build something that can explain itself, adapt to different investor profiles, and generate a complete research-driven portfolio.

Most current “robo-advisor” style LLM demos are extremely rigid and usually rely on generic templated logic. They also lack real explainability, which is a huge blocker for applying LLMs in serious fields like portfolio management.
This project tackles both issues head-on.


**What This System Does**

The app walks a user through the same onboarding process a real advisor would, automatically:

1. Investor Profiling (Conversational)

A dedicated profiling agent collects:

risk tolerance

investment horizon

experience level

preferred sectors

constraints

tickers of interest

investment amount

It keeps the conversation on track and asks follow-ups only when needed.

2. Ticker Universe Planning

Another agent selects a research universe using:

user watchlist (required)

diversification logic

sector balancing

ETF recommendations (intentionally emphasized)

It outputs strict JSON, so downstream logic never breaks.

3. Multi-Agent Research Pipeline

For every selected ticker:

Market data (returns, volatility, Sharpe-like ratio, trend signals)

News & sentiment (Finnhub API + Llama3 summarizer)

SEC Filings RAG

dynamic online 10-K scraping

token-window chunking

semantic + hybrid retrieval

investor-level summaries

Everything is stored per-ticker for clean handoff into the decision stage.

4. Portfolio Decision Agent

This is where everything comes together.

The decision agent:

categorizes each ticker into buckets (Core / Growth / Speculative)

assigns weights based on risk & research

generates clear rationale
(explicitly referencing market data, sentiment, and SEC findings)

The result is a human-readable portfolio that looks the way an actual analyst would structure it.

**Running the App**

Make sure Ollama is running and you have the models installed:

ollama run llama3
ollama run nomic-embed-text


Then start the Streamlit app:

streamlit run app.py


You’ll be greeted by the onboarding assistant, and once your profile is complete, you can generate the full investment strategy.

Project Structure
src/
  agents/
    crews/               # All multi-agent orchestration logic
    ticker_planner.py    # Universe planner (strict JSON)
    investment_strategy_crew.py
    portfolio_decision_crew.py
    profiling_crew.py
    models.py            # Pydantic user profile model

  rag/
    index_builder.py     # FAISS index builder (optional)
    retriever.py         # Semantic + hybrid RAG search
    answerer.py          # SEC-focused RAG answer engine

  ingestion/
    sec_loader.py        # Online 10-K scraping + chunking

app.py                   # Streamlit application
data/
  sec_index/             # Auto-generated FAISS index files

Requirements

All dependencies are listed in requirements.txt.
To install:

pip install -r requirements.txt

Environment Variables

Create a .env with:

FINNHUB_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here      # only needed if switching to OpenAI models


(These are already excluded in .gitignore.)

**Why I Built This**

Two main motivations:

LLMs need explainability if they’re going to touch real finance.
Black-box answers aren’t acceptable when investors need to understand risk.

Agentic systems are the future of AI workflows.
Instead of one giant model trying to guess everything, multiple specialized agents hand off structured outputs and build something far more reliable.

This project is my attempt at showing what a real, research-backed AI advisor could look like.