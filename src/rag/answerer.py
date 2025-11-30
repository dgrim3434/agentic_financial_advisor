from __future__ import annotations

from typing import List, Dict, Optional
from textwrap import dedent

import numpy as np
from ollama import Client as OllamaClient

from .retriever import (
    search_sec_filings,
    search_sec_filings_hybrid,
    embed_text,
)
from src.ingestion.sec_loader import load_and_chunk_filings  # online SEC loader

ollama = OllamaClient(host="http://localhost:11434")
LLM_MODEL = "llama3"


def _call_llm(prompt: str) -> str:
    """
    Basic wrapper around the Ollama chat endpoint.
    """
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": prompt},
        ],
    )
    # Ollama returns a dict
    return resp["message"]["content"]


def _format_context(chunks: List[Dict]) -> str:
    """
    Turn retrieved SEC chunks into a text block for the LLM.
    Keeps the metadata visible so we can reference DOC IDs later.
    """
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        company = meta.get("company", "UNKNOWN")
        filing_type = meta.get("filing_type", "UNKNOWN")
        year = meta.get("year", "UNKNOWN")
        src = meta.get("source_file", "UNKNOWN")

        lines.append(
            f"[DOC {i}] {company} {filing_type} {year} (source: {src})\n{c['text']}\n"
        )
    return "\n\n".join(lines)



# Online per-ticker search using sec_loader + embeddings
def _search_sec_filings_online(
    question: str,
    ticker: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    On-the-fly RAG over the latest filings fetched from the SEC website
    for a single ticker.

    Flow:
      1. load_and_chunk_filings([ticker]) -> list of docs with 'text' + metadata.
      2. Embed the question and each chunk.
      3. Rank with cosine similarity.
      4. Return top_k hits in the same shape as the FAISS-based search.
    """
    docs = load_and_chunk_filings([ticker])
    if not docs:
        return []

    # Question embedding
    q_emb = embed_text(question)
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        return []
    q_emb = q_emb / q_norm

    hits: List[Dict] = []

    for d in docs:
        text = d["text"]
        d_emb = embed_text(text)
        d_norm = np.linalg.norm(d_emb)
        if d_norm == 0:
            continue
        d_emb = d_emb / d_norm

        score = float(np.dot(q_emb, d_emb))  # cosine similarity

        meta = {k: v for k, v in d.items() if k != "text"}
        hits.append(
            {
                "id": meta.get("id"),
                "text": text,
                "metadata": meta,
                "score": score,
                "distance": float(max(0.0, 1.0 - score)),
            }
        )

    # Highest similarity first
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def answer_with_sec_rag(
    question: str,
    ticker: Optional[str] = None,
    use_hybrid: bool = True,
    top_k: int = 5,
    use_online_loader: bool = True,
) -> str:
    """
    High-level SEC RAG helper.

    Steps:
      1. Retrieve relevant SEC chunks:
         - If ticker is set and use_online_loader=True, fetch the latest filings
           dynamically via sec_loader.
         - Otherwise, fall back to the FAISS-based search.
      2. Call the LLM with those chunks and the user question.
      3. Return the LLM answer plus a short "Sources" section.
    """
    hits: List[Dict] = []

    # 1) Preferred path: online loader per ticker
    if use_online_loader and ticker is not None:
        try:
            hits = _search_sec_filings_online(
                question=question,
                ticker=ticker,
                top_k=top_k,
            )
        except Exception as e:
            
            print(f"[WARN] Online SEC loader failed for {ticker}: {e}")

    # 2) Fallback: existing FAISS-based RAG
    if not hits:
        if use_hybrid:
            hits = search_sec_filings_hybrid(
                query=question,
                company=ticker,
                top_k=top_k,
            )
        else:
            hits = search_sec_filings(
                query=question,
                company=ticker,
                top_k=top_k,
            )

    if not hits:
        return (
            "I couldn’t find any relevant SEC filing chunks for that query. "
            "Try a different question or ticker."
        )

    context_block = _format_context(hits)

    prompt = dedent(
        f"""
        You are an equity analyst specializing in SEC filings (10-K, 10-Q).

        The user question is:
        "{question}"

        Below are excerpts from official company filings that may be relevant:

        {context_block}

        Using ONLY the information in these excerpts:
        - Answer the user's question clearly and concisely.
        - Highlight key risks, business details, or disclosures if relevant.
        - If the answer is uncertain or not fully covered in the text, say so explicitly.
        - Do NOT hallucinate or fabricate specific numbers or filings that are not evident.

        Then, at the end, add a short "Sources" section listing which DOC IDs you used.
        """
    ).strip()

    answer = _call_llm(prompt)

    # Build a simple human-readable sources list for the tail of the response
    sources_lines = ["\n\nSources used:"]
    for i, h in enumerate(hits, start=1):
        meta = h["metadata"]
        sources_lines.append(
            f"[DOC {i}] {meta.get('company')} {meta.get('filing_type')} "
            f"{meta.get('year')} (source: {meta.get('source_file')})"
        )

    return answer + "\n" + "\n".join(sources_lines)

# Test
if __name__ == "__main__":
    
    q = "What risks does the company mention related to supply chain?"
    print("=== SEC RAG Answer Test ===\n")
    out = answer_with_sec_rag(
        question=q,
        ticker="GOOGL",
        use_hybrid=True,
        top_k=3,
        use_online_loader=True,
    )
    print(out)
