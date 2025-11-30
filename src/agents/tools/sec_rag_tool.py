from __future__ import annotations

from typing import List, Optional

from crewai.tools import tool
from ollama import Client as OllamaClient

from src.rag.answerer import answer_with_sec_rag

# Ollama client + model for summarization
_sec_ollama = OllamaClient(host="http://localhost:11434")
SEC_SUM_MODEL = "llama3"


def _normalize_tickers(raw: Optional[str]) -> List[str]:
    """Turn a comma/space-separated string of tickers into a clean list."""
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _summarize_sec_answer_for_investor(
    ticker: str,
    question: str,
    full_answer: str,
) -> str:
    """
    Use Ollama to turn a long SEC RAG answer into a very short,
    investor-focused summary for a single ticker.
    """
    # If the RAG step found nothing, surface that cleanly
    if "couldn’t find any relevant sec filing chunks" in full_answer.lower():
        return (
            f"### {ticker}\n"
            "- No sufficiently relevant SEC filing excerpts were retrieved for this query.\n"
        )

    # Build summarization prompt
    prompt = (
        f"You are an equity analyst specializing in SEC filings.\n\n"
        f"Ticker: {ticker}\n"
        f"Investor question or focus:\n{question}\n\n"
        "Below is a detailed answer generated from SEC filings excerpts. "
        "Your task is to summarize this answer for a long-term investor.\n\n"
        "Requirements:\n"
        "- Output ONLY 3–5 ultra-short bullet points.\n"
        "- Focus on the most important things an investor should know:\n"
        "  * core business drivers and revenue sources\n"
        "  * major long-term risks (competition, regulation, tech disruption, "
        "    leverage/liquidity)\n"
        "  * any key disclosures or red flags\n"
        "- Be concise and factual. No fluff. No numbers unless clearly supported "
        "  by the text.\n"
        "- Do NOT mention 'DOC IDs' or internal metadata; just the substance.\n\n"
        "Full SEC-based answer:\n"
        f"{full_answer}"
    )

    try:
        resp = _sec_ollama.chat(
            model=SEC_SUM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise SEC filings analyst. "
                        "Always respond in very short bullet points for investors."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = resp["message"]["content"].strip()
        if not content.startswith("-"):
            content = "- " + content.replace("\n", "\n- ")
        return f"### {ticker}\n{content}\n"
    except Exception as e:
        # Fallback: include first part of the original answer if LLM fails
        trimmed = full_answer.strip()
        if len(trimmed) > 800:
            trimmed = trimmed[:797] + "..."
        return (
            f"### {ticker}\n"
            "- LLM summarization failed; showing truncated SEC-based answer instead.\n\n"
            f"{trimmed}\n\n"
            f"- [Debug] LLM error: {e}\n"
        )


@tool("sec_filings_research_tool")
def sec_filings_research_tool(
    question: str,
    tickers: str = "",
) -> str:
    """
    Researches SEC filings (10-K / 10-Q, etc.) using your RAG pipeline,
    then compresses the result with an additional LLM into very short
    investor-focused summaries per ticker.

    Args:
        question:
            Natural language question about companies' risks, business models,
            or other fundamentals. This will be reused for each ticker.
        tickers:
            Comma-separated stock symbols (e.g. "AAPL, MSFT, NVDA").
            If empty, the RAG search will run across all indexed companies,
            and you'll get one combined summary.

    Returns:
        Markdown answer that:
        - Has '### TICKER' sections when tickers are specified.
        - For each ticker, contains ONLY 3–5 concise bullet points summarizing
          the key themes from the filings relevant to the question.
    """
    symbols = _normalize_tickers(tickers)

    # CASE 1: No specific ticker list — search across all companies and then summarize
    if not symbols:
        try:
            full_answer = answer_with_sec_rag(
                question=question,
                ticker=None,
                use_hybrid=True,
                top_k=5,
            )

            # Summarize the cross-company answer into short bullets
            prompt = (
                "You are an equity analyst summarizing SEC filings across multiple "
                "companies.\n\n"
                f"Investor question:\n{question}\n\n"
                "Below is a detailed answer based on SEC filings from many tickers. "
                "Summarize the key themes for a diversified long-term investor.\n\n"
                "Requirements:\n"
                "- Output ONLY 4–6 short bullet points.\n"
                "- Focus on broad themes: common risks, typical business drivers, "
                "  structural issues, or recurring disclosures.\n"
                "- No company-specific minutiae; think portfolio-level insights.\n\n"
                f"Full answer:\n{full_answer}"
            )

            try:
                resp = _sec_ollama.chat(
                    model=SEC_SUM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a concise SEC filings analyst. "
                                "Always respond in very short bullet points."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                content = resp["message"]["content"].strip()
                if not content.startswith("-"):
                    content = "- " + content.replace("\n", "\n- ")
                return (
                    "## SEC Filings Research (all indexed companies)\n\n"
                    f"**Question:** {question}\n\n"
                    f"{content}\n"
                )
            except Exception as e:
                return (
                    "## SEC Filings Research (all indexed companies)\n\n"
                    f"**Question:** {question}\n\n"
                    "LLM summarization failed; showing full SEC-based answer.\n\n"
                    f"{full_answer}\n\n"
                    f"- [Debug] LLM error: {e}\n"
                )

        except Exception as e:
            return (
                "SEC RAG tool encountered an error while answering your question.\n\n"
                f"Question: {question}\n"
                "Ticker scope: ALL\n"
                f"Error: {e}"
            )

    # CASE 2: Per-ticker RAG calls + summarization
    blocks: List[str] = []
    for sym in symbols:
        try:
            full_ans = answer_with_sec_rag(
                question=question,
                ticker=sym,
                use_hybrid=True,
                top_k=5,
            )
            summary_md = _summarize_sec_answer_for_investor(
                ticker=sym,
                question=question,
                full_answer=full_ans,
            )
            blocks.append(summary_md)
        except Exception as e:
            blocks.append(
                f"### {sym}\n\n"
                "SEC RAG tool encountered an error for this ticker.\n\n"
                f"- Question: {question}\n"
                f"- Error: {e}\n"
            )

    header = (
        "## SEC Filings Research\n\n"
        f"**Question:** {question}\n\n"
    )
    return header + "\n".join(blocks)

