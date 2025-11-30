from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
import os 
from src.agents.models import UserProfile


# LLM helper (OpenAI via LangChain)
def _get_openai_llm() -> ChatOpenAI:
    """
    Central OpenAI LLM config for the portfolio decision step.

    Assumes OPENAI_API_KEY is set in the environment
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )



# Build compact research context from research bundle



def _build_research_context(
    research_bundle: Dict[str, Any],
    max_chars_per_ticker: int = 800,
) -> str:
    """
    Compress the per-ticker research into a single context string
    that the decision agent can actually handle.

    Expected structure in research_bundle:
        {
          "universe": [...],
          "per_ticker": {
            "AAPL": { "market": "...", "sentiment": "...", "sec": "..." },
            ...
          }
        }

    For each ticker, we stitch together:
        - market notes
        - sentiment notes
        - SEC notes

    and then hard-truncate the combined text to max_chars_per_ticker to
    keep the prompt size under control.
    """
    universe: List[str] = research_bundle.get("universe", [])
    per_ticker: Dict[str, Dict[str, str]] = research_bundle.get("per_ticker", {})

    lines: List[str] = []

    for ticker in universe:
        data = per_ticker.get(ticker, {})
        if not data:
            continue

        market = (data.get("market") or "").strip()
        sentiment = (data.get("sentiment") or "").strip()
        sec = (data.get("sec") or "").strip()

        combined_parts = [p for p in [market, sentiment, sec] if p]
        if not combined_parts:
            continue

        combined = "\n\n".join(combined_parts).strip()

        # Aggressively truncate to keep the prompt small and predictable
        if len(combined) > max_chars_per_ticker:
            combined = combined[: max_chars_per_ticker - 3] + "..."

        lines.append(f"### {ticker}\n{combined}\n")

    return "\n".join(lines)



# Extract JSON block from LLM output



def _extract_json_block(raw: str) -> Dict[str, Any]:
    """
    Best-effort helper to pull a JSON object out of the model output.

    Strategy:
      - If it's already a dict, just return it.
      - Otherwise, grab the first {...} block and try to json.loads it.
      - On any failure, return {} so the caller can handle it.
    """
    if isinstance(raw, dict):
        return raw

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


# Main entry: portfolio decision agent


def run_portfolio_decision(
    profile: UserProfile,
    research_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Take the research output and turn it into a concrete portfolio plan.

    Inputs:
      - profile: UserProfile (risk tolerance, horizon, amount, etc.)
      - research_bundle: output of run_research_universe()

    Returns a decision_bundle shaped like:

      {
        "allocation_mode": "percent",
        "total_investment": 10000.0 or None,
        "positions": [
          {
            "ticker": "AAPL",
            "bucket": "core",
            "weight_pct": 25.0,
            "dollars": None,
            "rationale": "short thesis..."
          },
          ...
        ],
        "notes": "high-level notes",
        "narrative": "1–3 paragraph explanation of the overall strategy"
      }

    The Streamlit app is responsible for displaying this nicely.
    """
    universe: List[str] = research_bundle.get("universe", [])
    per_ticker: Dict[str, Dict[str, str]] = research_bundle.get("per_ticker", {})

    if not universe or not per_ticker:
        return {
            "allocation_mode": "percent",
            "total_investment": None,
            "positions": [],
            "notes": "No research available; cannot form a portfolio.",
            "narrative": (
                "The decision agent could not build a portfolio because "
                "no research universe was provided."
            ),
        }

    llm = _get_openai_llm()
    research_context = _build_research_context(research_bundle)
    profile_json = json.dumps(asdict(profile), indent=2)

    # Try to parse total investment if present in profile
    total_investment: Optional[float] = None
    if profile.investment_amount:
        try:
            total_investment = float(profile.investment_amount)
        except Exception:
            total_investment = None

    #Prompt

    system_prompt = (
        "You are a conservative, explainable **portfolio manager**.\n"
        "You receive:\n"
        "  1) A structured investor profile.\n"
        "  2) Summarized research for each ticker (market trends, news/sentiment, "
        "     and SEC fundamentals/risks).\n\n"
        "Your job is to propose a **diversified equity portfolio** using only these tickers.\n\n"
        "Buckets:\n"
        "  - 'core'        = stable, high-quality holdings that anchor the portfolio.\n"
        "  - 'growth'      = higher upside, reasonable risk.\n"
        "  - 'speculative' = high-risk positions; small combined weight.\n\n"
        "STRICT REQUIREMENTS FOR EACH TICKER YOU INCLUDE:\n"
        "  1. Decide whether the ticker is held as core, growth, or speculative.\n"
        "  2. Assign a weight_pct as a **relative score** (the system rescales to 100%).\n"
        "  3. The rationale MUST:\n"
        "       - Explain clearly WHY the ticker is included at all, AND\n"
        "       - Explain WHY its weight is higher/lower than other names "
        "         (e.g., 'core anchor so larger weight', 'speculative so capped at 5%').\n"
        "       - Explicitly reference evidence from ALL THREE research streams "
        "         whenever possible:\n"
        "           * Market/quant (trend, volatility, drawdowns, stability)\n"
        "           * News & sentiment (positive/negative themes, lack of news)\n"
        "           * SEC filings (fundamentals, long-term risks, leverage, regulation)\n"
        "         If one of the three streams provides little or no signal, you MUST say so "
        "         explicitly in the rationale (e.g., 'limited recent news; sentiment neutral').\n\n"
        "Other rules:\n"
        "- Align overall risk with the investor's risk tolerance and horizon.\n"
        "- Prefer a small set of well-justified positions.\n"
        "- Do NOT invent facts beyond the research summaries.\n"
        "- Each ticker may appear **at most once** in the positions array.\n"
        "- Output ONLY valid JSON, no commentary or markdown."
    )

    user_prompt = f"""
Investor profile (JSON):
{profile_json}

Research summaries by ticker:
{research_context}

Now construct a portfolio.

Return JSON of the form:

{{
  "allocation_mode": "percent",
  "total_investment": {total_investment if total_investment is not None else "null"},
  "positions": [
    {{
      "ticker": "AAPL",
      "bucket": "core" | "growth" | "speculative",
      "weight_pct": 25.0,  // treat this as a relative score; the system will rescale to sum to 100
      "rationale": "2–3 sentences that (a) justify including the stock, (b) justify this
                    relative weight (larger/smaller vs. others), and (c) explicitly reference
                    evidence from market behavior, news/sentiment, and SEC fundamentals/risks.
                    If one of those three streams had limited signal, say that explicitly."
    }},
    ...
  ],
  "notes": "brief comments about diversification and risk.",
  "narrative": "1–3 paragraph explanation of the overall strategy, including how the buckets
                (core/growth/speculative) and weights align with the investor's risk profile."
}}
"""

    resp = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    raw_answer = resp.content if hasattr(resp, "content") else str(resp)
    parsed = _extract_json_block(raw_answer)

    raw_positions = parsed.get("positions", []) or []

    # De-duplicate tickers:
    #   - merge weights
    #   - choose the MOST CONSERVATIVE bucket (core < growth < speculative)
    #   - glue rationales together so nothing is lost
    
    bucket_priority = {"core": 0, "growth": 1, "speculative": 2}

    merged_by_ticker: Dict[str, Dict[str, Any]] = {}

    for pos in raw_positions:
        try:
            ticker = str(pos.get("ticker", "")).upper().strip()
            if not ticker or ticker not in universe:
                continue

            # Bucket normalization with a simple fallback if the model gives junk
            bucket_raw = str(pos.get("bucket", "")).lower().strip()
            if bucket_raw not in bucket_priority:
                rt = (profile.risk_tolerance or "").lower()
                if rt.startswith("low"):
                    bucket_raw = "core"
                elif rt.startswith("high"):
                    bucket_raw = "speculative"
                else:
                    bucket_raw = "growth"

            weight_raw = float(pos.get("weight_pct", 0.0))
            if weight_raw <= 0:
                continue

            rationale = str(pos.get("rationale", "")).strip()

            if ticker not in merged_by_ticker:
                merged_by_ticker[ticker] = {
                    "ticker": ticker,
                    "bucket": bucket_raw,
                    "weight_raw": weight_raw,
                    "rationale": rationale,
                    "dollars": None,  # left None; the app can compute if needed
                }
            else:
                existing = merged_by_ticker[ticker]
                # Add up weights from duplicate entries
                existing["weight_raw"] += weight_raw

                # Keep the more conservative bucket if they conflict
                old_b = existing.get("bucket", "growth")
                if bucket_priority.get(bucket_raw, 1) < bucket_priority.get(old_b, 1):
                    existing["bucket"] = bucket_raw

                # Append additional rationale so nothing is thrown away
                if rationale:
                    if existing["rationale"]:
                        existing["rationale"] += " Additional reasoning: " + rationale
                    else:
                        existing["rationale"] = rationale

        except Exception:
            # If something looks off for one entry, just skip it and keep going
            continue

    clean_positions: List[Dict[str, Any]] = list(merged_by_ticker.values())

    # Rescale weights so they sum to ~100%
    raw_weights = [p["weight_raw"] for p in clean_positions]
    total_raw = sum(raw_weights)
    if total_raw > 0 and clean_positions:
        factor = 100.0 / total_raw
        for pos in clean_positions:
            pos["weight_pct"] = round(pos.pop("weight_raw") * factor, 2)
    else:
        clean_positions = []

    notes = parsed.get("notes", "")
    narrative = parsed.get(
        "narrative",
        "The decision agent produced a portfolio, but did not include a narrative explanation.",
    )

    return {
        "allocation_mode": "percent",
        "total_investment": total_investment,
        "positions": clean_positions,
        "notes": notes,
        "narrative": narrative,
    }



# Standalone debug


if __name__ == "__main__":
    # Quick local sanity check without going through Streamlit
    from src.agents.models import UserProfile  # local import for debug runs

    fake_profile = UserProfile(
        name="Debug User",
        age=25,
        risk_tolerance="medium",
        investment_horizon="10+ years",
        experience_level="beginner",
        preferred_sectors=["Technology", "Healthcare"],
        investment_amount="10000",
        ticker_watchlist=["AAPL", "MSFT", "GOOGL"],
        notes="Debug run.",
    )

    fake_research = {
        "universe": ["AAPL", "MSFT", "GOOGL"],
        "per_ticker": {
            "AAPL": {
                "market": "AAPL: steady uptrend, moderate volatility.",
                "sentiment": "Generally positive news around iPhone and services.",
                "sec": "Strong cash flow and buybacks; key risks in regulation and supply chain.",
            },
            "MSFT": {
                "market": "MSFT: strong long-term uptrend, low volatility.",
                "sentiment": "Positive AI and cloud coverage.",
                "sec": "Diversified revenue; risks in competition and regulatory scrutiny.",
            },
            "GOOGL": {
                "market": "GOOGL: growth with some drawdowns.",
                "sentiment": "Mixed sentiment around ad market and AI.",
                "sec": "Advertising dependency and antitrust investigations are key risks.",
            },
        },
    }

    bundle = run_portfolio_decision(fake_profile, fake_research)
    print(json.dumps(bundle, indent=2))
