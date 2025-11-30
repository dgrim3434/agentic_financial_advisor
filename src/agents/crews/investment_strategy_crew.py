from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from crewai import Agent, Crew, LLM, Task

from src.agents.models import UserProfile
from src.agents.ticker_planner import run_ticker_planner
from src.agents.tools.market_data_tool import market_data_tool
from src.agents.tools.news_sentiment_tool import news_sentiment_tool
from src.agents.tools.sec_rag_tool import sec_filings_research_tool



# LLM Helper function
def _get_ollama_llm() -> LLM:
    """
    Central place to configure the local Ollama LLM.

    If I ever change models or the base URL, I only need to update it here.
    """
    return LLM(
        model="ollama/llama3",
        base_url="http://localhost:11434",
    )


#String utilities
def _split_sections_by_ticker(text: str, tickers: List[str]) -> Dict[str, str]:
    """
    Take a long markdown string and break it into per-ticker sections.

    Assumes the text uses headers like:

        ### AAPL
        ...notes...

        ### MSFT
        ...notes...

    Returns a mapping like:
        {
          "AAPL": "...AAPL section...",
          "MSFT": "...MSFT section...",
          ...
        }

    If a ticker doesn’t appear, it just gets an empty string.
    """
    result: Dict[str, str] = {t: "" for t in tickers}
    if not text:
        return result

    # Find where each ticker header starts
    positions: List[Tuple[int, str]] = []
    for t in tickers:
        header = f"### {t}"
        idx = text.find(header)
        if idx != -1:
            positions.append((idx, t))

    if not positions:
        # No recognizable headers; return the empty structure
        return result

    # Process headers in the order they appear in the text
    positions.sort(key=lambda x: x[0])

    # Slice from each header up to the next one
    for i, (start_idx, ticker) in enumerate(positions):
        end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        block = text[start_idx:end_idx].strip()
        result[ticker] = block

    return result

#String utilities
def _task_output_to_str(output: Any) -> str:
    """
    Normalize a Task's output into a plain string.

    CrewAI can return a TaskOutput object or a bare string; for logging and
    downstream parsing it’s easier if we always deal with `str`.
    """
    if output is None:
        return ""
    try:
        # TaskOutput might expose .raw or .value, but str() is the safest
        return str(output)
    except Exception:
        return ""


# Agent builders
def _build_market_agent() -> Agent:
    llm = _get_ollama_llm()
    return Agent(
        role="Market & Quant Researcher",
        goal=(
            "Analyze historical price behavior, volatility, and basic risk/return "
            "signals for a list of tickers."
        ),
        backstory=(
            "You specialize in turning raw historical market data into clear, "
            "plain-language descriptions of how each ticker has been behaving."
        ),
        llm=llm,
        tools=[market_data_tool],
        verbose=False,
        allow_delegation=False,
    )

# Agent builders
def _build_sentiment_agent() -> Agent:
    llm = _get_ollama_llm()
    return Agent(
        role="News & Sentiment Researcher",
        goal=(
            "Summarize recent company news and sentiment for a list of tickers, "
            "highlighting bullish and bearish themes."
        ),
        backstory=(
            "You monitor headlines, company-specific news, and narrative shifts "
            "in the market. You boil this down into structured summaries that "
            "are easy for investors to read."
        ),
        llm=llm,
        tools=[news_sentiment_tool],
        verbose=True,  # keep tool traces for now
        allow_delegation=False,
    )

# Agent builders
def _build_sec_question_agent() -> Agent:
    llm = _get_ollama_llm()
    return Agent(
        role="SEC Research Question Designer",
        goal=(
            "Take market behavior notes and sentiment notes for a list of tickers "
            "and design focused guidance for what the SEC RAG pipeline should "
            "look for in filings."
        ),
        backstory=(
            "You are an expert in reading signals from price action and news, "
            "and turning them into concrete questions for 10-K and 10-Q filings."
        ),
        llm=llm,
        tools=[],
        verbose=False,
        allow_delegation=False,
    )

# Agent builders
def _build_sec_rag_agent() -> Agent:
    llm = _get_ollama_llm()
    return Agent(
        role="SEC Filings RAG Researcher",
        goal=(
            "Use a retrieval-augmented pipeline over SEC filings to extract "
            "fundamentals and long-term risk factors for each ticker."
        ),
        backstory=(
            "You know how to query a specialized SEC RAG tool to pull the most "
            "relevant paragraphs from recent 10-K and 10-Q filings and summarize "
            "them for investors."
        ),
        llm=llm,
        tools=[sec_filings_research_tool],
        verbose=False,
        allow_delegation=False,
    )

# Research Pipeline
def run_research_universe(profile: UserProfile) -> Dict[str, Any]:
    """
    Run the full research pass over the ticker universe for this user.

    High level flow:
      1. Use the ticker planner to decide the full universe of tickers.
      2. Run a 4-agent Crew on that universe:
            - Market & Quant agent (market_data_tool)
            - News & Sentiment agent (news_sentiment_tool)
            - SEC Question Designer agent (no tools)
            - SEC RAG agent (sec_filings_research_tool)
      3. Split each agent’s output into per-ticker blocks so it’s easy to use later.

    Returns a dict shaped like:

      {
        "plan": { ... ticker planner JSON ... },
        "universe": ["AAPL", "MSFT", ...],
        "tickers_csv": "AAPL,MSFT,...",

        "market_notes_raw": "...full markdown from market_data_tool...",
        "sentiment_notes_raw": "...full markdown from news_sentiment_tool...",
        "sec_focus_notes": "...markdown bullet list...",
        "sec_summary_raw": "...full markdown from sec_filings_research_tool...",

        "per_ticker": {
          "AAPL": {
            "market": "...AAPL section from market notes...",
            "sentiment": "...AAPL section from sentiment notes...",
            "sec": "...AAPL section from SEC summary..."
          },
          ...
        }
      }

    This function only handles the research. Any actual portfolio construction
    (weights, buckets, etc.) is done by a separate module on top of this output.
    """

    # Step 1: Decide the ticker universe using the planner
    plan: Dict[str, List[str]] = run_ticker_planner(profile)

    universe: List[str] = []
    for key in (
        "primary_tickers",
        "similar_tickers",
        "balancing_tickers",
        "other_tickers",
    ):
        for t in plan.get(key, []):
            t_up = str(t).upper()
            if t_up not in universe:
                universe.append(t_up)

    if not universe:
        # No tickers to look at, return a consistent empty structure
        return {
            "plan": plan,
            "universe": [],
            "tickers_csv": "",
            "market_notes_raw": "",
            "sentiment_notes_raw": "",
            "sec_focus_notes": "",
            "sec_summary_raw": "",
            "per_ticker": {},
        }

    tickers_csv = ",".join(universe)
    profile_json = json.dumps(asdict(profile), indent=2)

    # Step 2: Build agents & tasks
    market_agent = _build_market_agent()
    sentiment_agent = _build_sentiment_agent()
    sec_q_agent = _build_sec_question_agent()
    sec_rag_agent = _build_sec_rag_agent()

    # Market task
    market_task = Task(
        description=(
            "You are the Market & Quant Researcher.\n\n"
            "Investor profile JSON:\n"
            "{profile_json}\n\n"
            "Tickers to analyze (comma-separated): {tickers_csv}\n\n"
            "You MUST use the tool `market_data_tool` exactly once with:\n"
            '  tickers = the full comma-separated list (e.g. "AAPL,MSFT,GOOG")\n\n'
            "The tool signature is:\n"
            "  market_data_tool(tickers: str) -> str\n\n"
            "Use the tool's result to produce markdown with sections for each ticker:\n"
            "  ### TICKER\n"
            "  - Trend: ...\n"
            "  - Volatility: ...\n"
            "  - 2–4 key observations relevant to the investor's risk profile.\n"
            "Important hard rules:\n"
            "  - Never call `market_data_tool` more than once.\n"
            "  - After the first successful call, DO NOT issue any further"
        ),
        expected_output="Markdown with '### TICKER' sections summarizing market behavior.",
        agent=market_agent,
    )

    # Sentiment task
    sentiment_task = Task(
        description=(
            "You are the News & Sentiment Researcher.\n\n"
            "Investor profile JSON:\n{profile_json}\n\n"
            "Tickers to analyze (comma-separated): {tickers_csv}\n\n"
            "You MUST follow this procedure exactly:\n"
            "1. Call `news_sentiment_tool` EXACTLY ONCE with:\n"
            '   - tickers   = the full comma-separated list (e.g. "AAPL,MSFT,GOOGL")\n'
            "   - days_back = 7\n\n"
            "2. Wait for the tool result. It will return markdown with:\n"
            "   - A heading like '## News & Sentiment Snapshot (last N days)'\n"
            "   - One '### TICKER' section per ticker, with 2–3 short bullets each.\n\n"
            "3. Using ONLY the information from that tool output, write your final "
            "answer in THIS EXACT FORMAT and for EVERY ticker in this list:\n"
            "   {tickers_csv}\n\n"
            "Your output MUST look like:\n\n"
            "## News & Sentiment Snapshot (last 7 days)\n"
            "_Source: Finnhub company-news API + LLM summarization_\n\n"
            "### AAPL\n"
            "- Overall sentiment: positive / neutral / negative\n"
            "- Summary:\n"
            "  - short bullet summarizing the most important theme from the news\n"
            "  - another short bullet if there is a second important theme\n"
            "  - optional third bullet ONLY if it is clearly important\n\n"
            "### NEXT_TICKER\n"
            "(repeat the same structure for EVERY ticker in {tickers_csv}, "
            "in the SAME ORDER):\n"
            "- If the tool returned only generic or minor items, use "
            "  'Overall sentiment: neutral' and bullets like "
            "  'No material company-specific news; mostly routine or sector-wide updates.'\n"
            "- Never leave the 'Summary' section blank.\n\n"
            "Hard rules:\n"
            "- Do NOT group multiple tickers together; each ticker gets its own '### TICKER' section.\n"
            "- Do NOT invent headlines or stories; your bullets must be grounded in the tool's bullets.\n"
            "- Do NOT call `news_sentiment_tool` more than once.\n"
            "- After the first successful tool call, do NOT emit any further 'Action:' blocks; "
            "just write the final markdown in the format above.\n"
        ),
        expected_output=(
            "Markdown starting with '## News & Sentiment Snapshot (last 7 days)' and then "
            "one '### TICKER' block per ticker in the input list, each with an overall "
            "sentiment and a short bullet Summary."
        ),
        agent=sentiment_agent,
    )

    # SEC Question Designer task
    sec_q_task = Task(
        description=(
            "You are the SEC Research Question Designer.\n\n"
            "Investor profile JSON:\n"
            "{profile_json}\n\n"
            "Tickers under consideration (comma-separated): {tickers_csv}\n\n"
            "You will see, in previous context messages:\n"
            "- The Market & Quant research markdown (per-ticker sections)\n"
            "- The News & Sentiment research markdown (per-ticker sections)\n\n"
            "Your job is to write a section titled 'SEC Research Focus' that contains "
            "bullet points describing what a downstream SEC RAG pipeline should look "
            "for in recent 10-K and 10-Q filings for EACH ticker.\n\n"
            "Focus especially on:\n"
            "- Revenue drivers and business model\n"
            "- Competitive and technology risks\n"
            "- Regulation and legal risks\n"
            "- Profitability, margins, cash flow, and leverage\n"
            "- Any risks or themes raised by the market or news.\n"
        ),
        expected_output="A markdown section titled 'SEC Research Focus' with bullet points.",
        agent=sec_q_agent,
        context=[market_task, sentiment_task],
    )

    # SEC RAG task
    sec_rag_task = Task(
        description=(
            "You are the SEC Filings RAG Researcher.\n\n"
            "Investor profile JSON:\n"
            "{profile_json}\n\n"
            "Tickers to analyze (comma-separated): {tickers_csv}\n\n"
            "You will see, in previous context messages, a section titled "
            "'SEC Research Focus' that lists the key topics the RAG system should "
            "look for in the filings.\n\n"
            "You MUST call `sec_filings_research_tool` exactly once with:\n"
            '  tickers  = the full comma-separated list (e.g. "AAPL,MSFT,GOOG")\n'
            "  question = a detailed natural-language query that includes the "
            "             'SEC Research Focus' guidance.\n\n"
            "The tool signature is:\n"
            "  sec_filings_research_tool(tickers: str, question: str) -> str\n\n"
            "Use the tool's response to produce markdown with sections for each ticker:\n"
            "  ### TICKER\n"
            "  - Fundamentals overview (1–2 paragraphs)\n"
            "  - 3–6 long-term risk factors\n"
            "  - Any disclosures that connect directly to concerns from market/news.\n"
        ),
        expected_output="Markdown with '### TICKER' sections summarizing SEC filings insights.",
        agent=sec_rag_agent,
        context=[sec_q_task],
    )

    # Step 3: Run the Crew sequentially with shared inputs
    crew = Crew(
        agents=[market_agent, sentiment_agent, sec_q_agent, sec_rag_agent],
        tasks=[market_task, sentiment_task, sec_q_task, sec_rag_task],
        verbose=False,
        process="sequential",
    )

    inputs = {
        "profile_json": profile_json,
        "tickers_csv": tickers_csv,
    }

    result = crew.kickoff(inputs=inputs)

    # Normalize raw outputs from tasks
    market_notes_raw = _task_output_to_str(market_task.output)
    sentiment_notes_raw = _task_output_to_str(sentiment_task.output)
    sec_focus_notes = _task_output_to_str(sec_q_task.output)
    sec_summary_raw = _task_output_to_str(
        sec_rag_task.output or getattr(result, "output", None)
    )

    # Step 4: Build per-ticker structure
    market_by_ticker = _split_sections_by_ticker(market_notes_raw, universe)
    sentiment_by_ticker = _split_sections_by_ticker(sentiment_notes_raw, universe)
    sec_by_ticker = _split_sections_by_ticker(sec_summary_raw, universe)

    per_ticker: Dict[str, Dict[str, str]] = {}
    for t in universe:
        per_ticker[t] = {
            "market": market_by_ticker.get(t, ""),
            "sentiment": sentiment_by_ticker.get(t, ""),
            "sec": sec_by_ticker.get(t, ""),
        }

    return {
        "plan": plan,
        "universe": universe,
        "tickers_csv": tickers_csv,
        "market_notes_raw": market_notes_raw,
        "sentiment_notes_raw": sentiment_notes_raw,
        "sec_focus_notes": sec_focus_notes,
        "sec_summary_raw": sec_summary_raw,
        "per_ticker": per_ticker,
    }


# Investment_strategy_crew Debugging point
if __name__ == "__main__":
    """
    Quick CLI sanity check so I can run this module without Streamlit:

        python -m src.agents.crews.investment_strategy_crew
    """
    profile = UserProfile(
        name="Debug User",
        age=22,
        risk_tolerance="medium",
        investment_horizon="10+ years",
        experience_level="beginner",
        preferred_sectors=["Technology", "AI"],
        investment_amount="10000",
        ticker_watchlist=["AAPL"],  # starting point for planner
        notes="Standalone debug run.",
    )

    out = run_research_universe(profile)

    print("\n" + "#" * 80)
    print("[DEBUG] Ticker Plan:")
    print(json.dumps(out["plan"], indent=2))

    print("\n[DEBUG] Universe:", out["universe"])
    print("#" * 80 + "\n")

    print("[DEBUG] Market notes length:", len(out["market_notes_raw"]))
    print("[DEBUG] Sentiment notes length:", len(out["sentiment_notes_raw"]))
    print("[DEBUG] SEC focus notes length:", len(out["sec_focus_notes"]))
    print("[DEBUG] SEC summary notes length:", len(out["sec_summary_raw"]))

    print("\n[DEBUG] Example per_ticker entry for first ticker:")
    if out["universe"]:
        first = out["universe"][0]
        print(f"[DEBUG] First ticker: {first}")
        print(json.dumps(out["per_ticker"].get(first, {}), indent=2))
    else:
        print("[DEBUG] Universe is empty.")
