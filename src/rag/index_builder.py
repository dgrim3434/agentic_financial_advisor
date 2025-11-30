from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import json

import numpy as np
import faiss
from ollama import Client

from src.ingestion.sec_loader import load_and_chunk_filings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the FAISS index + metadata are stored
INDEX_DIR = PROJECT_ROOT / "data" / "sec_index"
INDEX_PATH = INDEX_DIR / "sec_faiss.index"
META_PATH = INDEX_DIR / "sec_metadata.json"

EMBED_MODEL = "nomic-embed-text"
ollama = Client(host="http://localhost:11434")


def embed_text(text: str) -> List[float]:
    """
    Get an embedding vector from Ollama (nomic-embed-text) for a given string.
    """
    resp = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text,
    )
    return resp["embedding"]


def build_index(tickers: List[str]) -> None:
    """
    Build a FAISS index over SEC filing chunks for the given tickers.

    This uses the online sec_loader, which pulls filings from EDGAR,
    chunks them, and returns a list of dicts with text + metadata.
    """
    print(f"[INFO] Loading and chunking SEC filings for: {tickers}")
    docs = load_and_chunk_filings(tickers)

    if not docs:
        print("[ERROR] No documents returned from load_and_chunk_filings.")
        return

    print(f"[INFO] Loaded {len(docs)} chunks in index_builder.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embeddings: List[List[float]] = []
    metadatas: List[Dict] = []
    texts: List[str] = []

    for d in docs:
        text = d["text"]
        emb = embed_text(text)

        embeddings.append(emb)
        texts.append(text)
        metadatas.append(
            {
                "id": d["id"],
                "company": d["company"],
                "filing_type": d["filing_type"],
                "year": d["year"],
                "quarter": d["quarter"],
                "source_file": d["source_file"],
            }
        )

    # Convert embeddings to numpy array
    X = np.array(embeddings, dtype="float32")
    if X.ndim != 2:
        raise RuntimeError(f"[ERROR] Embedding matrix has wrong shape: {X.shape}")

    dim = X.shape[1]
    print(f"[INFO] Building FAISS index with dim={dim}, n_vectors={X.shape[0]}")

    # Use inner-product index with normalized vectors
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[INFO] Saved FAISS index to {INDEX_PATH}")

    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": texts,
                "metadatas": metadatas,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved metadata to {META_PATH}")
    print("[SUCCESS] SEC FAISS index build complete.")

# Test
if __name__ == "__main__":
    DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
    build_index(DEFAULT_TICKERS)
