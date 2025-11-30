from __future__ import annotations

import textwrap
from datetime import datetime, timedelta
from typing import List

import requests
from crewai.tools import tool
from ollama import Client as OllamaClient



# Config: Finnhub + local Ollama model for news summarization


# Finnhub API key is expected to be defined somewhere upstream
FINNHUB_API_KEY = FINNHUB_API_KEY

# Local LLM used to compress raw news into short investor summaries
_ollama = OllamaClient(host="http://localhost:11434")
NEWS_SUM_MODEL = "llama3"



# Finnhub helper

def _fetch_company_news(symbol: str, days_back: int) -> List[dict]:
    """
    Hit Finnhub's company-news endpoint for a single symbol.

    Returns:
        A list of raw article dicts (or [] on error / no data).
    """
    if not FINNHUB_API_KEY:
        return []

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        # If this fails, we just fall back to a "no news" summary later
        return []



# LLM-side summarization helpers

def _build_article_block(articles: List[dict]) -> str:
    """
    Turn raw Finnhub articles into a compact plain-text block
    for the LLM to summarize.
    """
    lines: List[str] = []

    # Cap articles per ticker to keep the prompt under control
    for art in articles[:12]:
        headline = str(art.get("headline", "")).strip() or "Untitled article"
        source = str(art.get("source", "")).strip()
        summary = str(art.get("summary", "")).strip()
        dt = art.get("datetime")
        url = str(art.get("url", "")).strip()

        parts = [f"Headline: {headline}"]
        if source:
            parts.append(f"Source: {source}")
        if dt:
            try:
                # Finnhub datetime is usually a Unix timestamp (seconds)
                dt_str = datetime.utcfromtimestamp(int(dt)).strftime("%Y-%m-%d")
                parts.append(f"Date: {dt_str}")
            except Exception:
                pass
        if summary:
            # Keep summaries but trim overly long ones
            if len(summary) > 400:
                summary = summary[:397] + "..."
            parts.append(f"Summary: {summary}")
        if url:
            parts.append(f"URL: {url}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)


def _summarize_news_for_investor(
    symbol: str,
    articles: List[dict],
    days_back: int,
) -> str:
    """
    Use the local Ollama model to write an ultra-short, investor-focused
    summary of recent news for a single ticker.
    """
    if not articles:
        return (
            f"### {symbol}\n"
            "- No meaningful recent company-specific news found in this period, "
            "or the news API did not return usable results.\n"
        )

    articles_block = _build_article_block(articles)

    prompt = textwrap.dedent(
        f"""
        You are an equity analyst writing for a long-term investor.

        Company ticker: {symbol}
        Lookback window: last {days_back} days.

        Below is a list of recent news items for this company. Each line may
        include headline, source, date, a short summary, and sometimes a URL.

        Your task:
        - Summarize these news items in the SHORTEST way possible.
        - Output ONLY 2–3 bullet points.
        - Focus ONLY on the most important themes or developments an investor
          should know (growth drivers, major risks, regulation, big catalysts,
          or anything materially affecting the business or stock).
        - Ignore minor or repetitive items.
        - Be concise, factual, and neutral. No hype.

        News items:
        {articles_block}
        """
    ).strip()

    try:
        resp = _ollama.chat(
            model=NEWS_SUM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise investment research assistant. "
                        "Always answer in bullet points, extremely briefly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = resp["message"]["content"].strip()

        # Make sure the response is bullet-pointed and wrapped under a ticker heading
        if not content.startswith("-"):
            content = "- " + content.replace("\n", "\n- ")

        return f"### {symbol}\n{content}\n"

    except Exception as e:
        # Fallback: simple non-LLM summary
        lines = [f"### {symbol}", ""]
        if not articles:
            lines.append(
                "- No meaningful recent company-specific news found or API call failed."
            )
            return "\n".join(lines) + "\n"

        lines.append("- Key recent headlines (LLM summarization failed):")
        for art in articles[:3]:
            headline = str(art.get("headline", "")).strip() or "Untitled article"
            source = str(art.get("source", "")).strip()
            if source:
                lines.append(f"  - {headline} ({source})")
            else:
                lines.append(f"  - {headline}")
        lines.append(f"  - [Debug] LLM error: {e}")
        return "\n".join(lines) + "\n"



# Public CrewAI tool

@tool("news_sentiment_tool")
def news_sentiment_tool(tickers: str, days_back: int = 7) -> str:
    """
    Fetch recent company news for one or more tickers (comma-separated)
    and return a VERY SHORT investor-focused markdown summary per ticker.

    Args:
        tickers:
            Comma-separated list of ticker symbols, e.g. "AAPL,MSFT,GOOGL".
        days_back:
            How many days back to request news for.

    Returns:
        Markdown string that starts with a global heading and then one
        '### TICKER' section per symbol in the input list (in order),
        each containing only a few concise bullet points.
    """
    # Normalize and dedupe tickers while preserving order
    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    seen = set()
    ordered_symbols: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered_symbols.append(s)

    lines: List[str] = []
    lines.append(f"## News & Sentiment Snapshot (last {days_back} days)")
    lines.append("_Source: Finnhub company-news API + LLM summarization_")
    lines.append("")

    if not ordered_symbols:
        lines.append(
            "- No valid tickers were provided. "
            "Expected a comma-separated string like 'AAPL,MSFT,GOOGL'."
        )
        return "\n".join(lines)

    for symbol in ordered_symbols:
        articles = _fetch_company_news(symbol, days_back)
        section_md = _summarize_news_for_investor(symbol, articles, days_back)
        lines.append(section_md)
        lines.append("")  # blank line between tickers

    return "\n".join(lines)
