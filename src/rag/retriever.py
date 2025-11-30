from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import json
import re
from collections import Counter

import numpy as np
import faiss
from ollama import Client as OllamaClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "sec_index"
INDEX_PATH = INDEX_DIR / "sec_faiss.index"
META_PATH = INDEX_DIR / "sec_metadata.json"

EMBED_MODEL = "nomic-embed-text"
ollama = OllamaClient(host="http://localhost:11434")



# Embeddings

def embed_text(text: str) -> np.ndarray:
    """
    Get an embedding vector for a string using Ollama (nomic-embed-text).
    """
    resp = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text,
    )
    emb = np.array(resp["embedding"], dtype="float32")
    return emb


def _load_index_and_meta():
    """
    Load FAISS index + metadata (texts + metadatas list).
    """
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            "SEC FAISS index or metadata not found. "
            "Run `python -m src.rag.index_builder` first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    texts: List[str] = meta["texts"]
    metadatas: List[Dict] = meta["metadatas"]

    if len(texts) != len(metadatas):
        raise RuntimeError(
            f"Metadata length mismatch: texts={len(texts)} metadatas={len(metadatas)}"
        )

    return index, texts, metadatas



# Simple lexical / hybrid helpers

STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "on",
    "at",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "as",
    "by",
    "with",
    "that",
    "this",
    "it",
    "from",
    "be",
    "or",
    "we",
    "our",
    "their",
    "they",
    "its",
    "has",
    "have",
}


def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: lowercase, split on non-words, drop stopwords.
    """
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def lexical_score(query_tokens: List[str], text: str) -> int:
    """
    Just counts how often the query tokens show up in the text.
    """
    text_tokens = tokenize(text)
    counts = Counter(text_tokens)
    return sum(counts[t] for t in query_tokens)



# semantic FAISS search 
def search_sec_filing_semantic(
    query: str,
    company: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict]:
    """
    Semantic search over SEC filings using FAISS (cosine via inner product).

    Returns list of:
      {
        "id": str,
        "text": str,
        "metadata": dict,
        "score": float,
        "distance": float,   # 1 - score (kept for backwards compatibility)
      }
    """
    index, texts, metadatas = _load_index_and_meta()

    q_emb = embed_text(query)
    q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)

    # Over-fetch so we can filter by company and still have top_k left
    n_search = max(top_k * 3, 10)
    scores, indices = index.search(q_emb, n_search)

    hits: List[Dict] = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        meta = metadatas[idx]
        text = texts[idx]

        if company is not None and meta["company"].upper() != company.upper():
            continue

        score_f = float(score)
        hits.append(
            {
                "id": meta["id"],
                "text": text,
                "metadata": meta,
                "score": score_f,
                "distance": float(max(0.0, 1.0 - score_f)),
            }
        )

        if len(hits) >= top_k:
            break

    return hits


def search_sec_filings(
    query: str,
    company: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict]:
    """
    Thin wrapper so older code can call search_sec_filings()
    and still get semantic-only behavior.
    """
    return search_sec_filing_semantic(query, company=company, top_k=top_k)



# Hybrid search: semantic + lexical

def search_sec_filings_hybrid(
    query: str,
    company: Optional[str] = None,
    top_k: int = 5,
    alpha: float = 0.6,
    semantic_k: Optional[int] = None,
) -> List[Dict]:
    """
    Hybrid search combining:
      - semantic similarity from FAISS
      - simple lexical overlap

    alpha controls how much weight semantic vs lexical scoring gets.
    """
    index, texts, metadatas = _load_index_and_meta()

    if semantic_k is None:
        semantic_k = max(top_k, 10)

    # 1) Semantic hits
    q_emb = embed_text(query).reshape(1, -1)
    faiss.normalize_L2(q_emb)

    s_scores, s_indices = index.search(q_emb, semantic_k)
    sem_hits: List[Dict] = []
    sem_scores: Dict[int, float] = {}

    for score, idx in zip(s_scores[0], s_indices[0]):
        if idx == -1:
            continue

        meta = metadatas[idx]
        text = texts[idx]

        if company is not None and meta["company"].upper() != company.upper():
            continue

        score_f = float(score)
        sem_scores[idx] = score_f
        sem_hits.append(
            {
                "idx": idx,
                "id": meta["id"],
                "text": text,
                "metadata": meta,
                "score": score_f,
                "distance": float(max(0.0, 1.0 - score_f)),
            }
        )

    # 2) Lexical scores across all docs (or filtered by company)
    query_tokens = tokenize(query)
    lex_scores: Dict[int, float] = {}

    for i, (text, meta) in enumerate(zip(texts, metadatas)):
        if company is not None and meta["company"].upper() != company.upper():
            continue
        score = lexical_score(query_tokens, text)
        if score > 0:
            lex_scores[i] = float(score)

    # 3) Normalize scores into [0, 1]
    if sem_scores:
        max_sem = max(sem_scores.values())
    else:
        max_sem = 1.0

    sem_norm = {i: (s / max_sem if max_sem > 0 else 0.0) for i, s in sem_scores.items()}

    if lex_scores:
        max_lex = max(lex_scores.values())
    else:
        max_lex = 1.0

    lex_norm = {i: (s / max_lex if max_lex > 0 else 0.0) for i, s in lex_scores.items()}

    # 4) Combine and rank
    combined: List[Dict] = []
    all_indices = set(list(sem_norm.keys()) + list(lex_norm.keys()))

    for i in all_indices:
        sem_s = sem_norm.get(i, 0.0)
        lex_s = lex_norm.get(i, 0.0)
        hybrid = alpha * sem_s + (1.0 - alpha) * lex_s
        if hybrid <= 0.0:
            continue

        meta = metadatas[i]
        text = texts[i]

        combined.append(
            {
                "id": meta["id"],
                "text": text,
                "metadata": meta,
                "semantic_score": float(sem_s),
                "lexical_score": float(lex_s),
                "hybrid_score": float(hybrid),
                "distance": float(max(0.0, 1.0 - sem_norm.get(i, 0.0))),
            }
        )

    combined.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return combined[:top_k]



# Test
if __name__ == "__main__":
    q = "What risks does the company mention related to supply chain?"

    print("=== Semantic only ===")
    sem_results = search_sec_filing_semantic(q, company="AAPL", top_k=3)
    for i, r in enumerate(sem_results, start=1):
        meta = r["metadata"]
        print(
            f"[{i}] {meta.get('company')} {meta.get('filing_type')} "
            f"{meta.get('year')} (source: {meta.get('source_file')})  "
            f"score={r['score']:.4f} dist={r['distance']:.4f}"
        )
        print(r["text"][:250], "...\n")

    print("\n=== Hybrid (semantic + lexical) ===")
    hyb_results = search_sec_filings_hybrid(q, company="AAPL", top_k=3)
    for i, r in enumerate(hyb_results, start=1):
        meta = r["metadata"]
        print(
            f"[{i}] {meta.get('company')} {meta.get('filing_type')} "
            f"{meta.get('year')} (source: {meta.get('source_file')})  "
            f"hybrid={r['hybrid_score']:.4f} "
            f"(sem={r['semantic_score']:.3f}, lex={r['lexical_score']:.3f})"
        )
        print(r["text"][:250], "...\n")
