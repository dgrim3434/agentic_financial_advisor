from __future__ import annotations

import re
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
import tiktoken


# Tokenization config for chunking SEC filings

ENCODING_NAME = "cl100k_base"
CHUNK_TOKENS = 800
CHUNK_OVERLAP = 200

EDGAR_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_USER_AGENT = {
    # Required by SEC; use a real name + email here
    "User-Agent": "YOUR_NAME, YOUR_EMAIL",
}


# Tokenizer + chunking helpers

def get_tokenizer():
    """Return the tiktoken encoding we use for splitting into token chunks."""
    return tiktoken.get_encoding(ENCODING_NAME)


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping token windows so chunks stay LLM-friendly.

    Example:
        - chunk_size = 800 tokens
        - overlap = 200 tokens
    """
    enc = get_tokenizer()
    tokens = enc.encode(text)

    chunks: List[str] = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_tokens = enc.decode(tokens[start:end])
        chunks.append(chunk_tokens)

        if end == n:
            break

        # Slide the window forward with overlap
        start = end - overlap

    return chunks



# Basic cleaning

def clean_text(text: str) -> str:
    """Collapse whitespace and trim."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()



# SEC scraping utilities

def _find_latest_10k_url(ticker: str) -> str:
    """
    Find the URL for the most recent 10-K filing HTML page for a ticker.
    """
    params = {
        "action": "getcompany",
        "CIK": ticker,
        "type": "10-K",
        "owner": "exclude",
        "count": "10",
    }

    r = requests.get(EDGAR_SEARCH_URL, params=params, headers=EDGAR_USER_AGENT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("table.tableFile2 tr")

    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 2:
            filing_type = cols[0].text.strip()
            if filing_type.upper() == "10-K":
                link = cols[1].find("a")["href"]
                return "https://www.sec.gov" + link

    raise RuntimeError(f"No 10-K filings found for {ticker}")


def _extract_full_10k_text(filing_page_url: str) -> str:
    """
    Given a 10-K filing page, grab the main HTML document and extract its text.
    """
    r = requests.get(filing_page_url, headers=EDGAR_USER_AGENT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Find the main filing document link
    doc_table = soup.find("table", class_="tableFile")
    if not doc_table:
        raise RuntimeError("Cannot locate filing documents table")

    for row in doc_table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 3:
            doc_type = cols[3].text.strip().upper() if len(cols) >= 4 else ""
            if doc_type == "10-K":
                doc_link = cols[2].find("a")["href"]
                full_url = "https://www.sec.gov" + doc_link
                return _download_html_text(full_url)

    # Fallback: just use whatever text is on the page we already have
    return _download_html_text(filing_page_url)


def _download_html_text(url: str) -> str:
    """
    Download HTML from a URL and return cleaned plain text.
    """
    r = requests.get(url, headers=EDGAR_USER_AGENT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Strip out scripts and styles
    for tag in soup(["script", "style"]):
        tag.extract()

    text = soup.get_text(separator=" ")
    return clean_text(text)



# Load + chunk latest 10-Ks

def load_and_chunk_filings(tickers: List[str]) -> List[Dict]:
    """
    Online loader for SEC filings used by the RAG pipeline.

    For each ticker:
      - Fetch the latest 10-K from EDGAR
      - Convert the HTML to plain text
      - Chunk into token windows
      - Return a list of dicts in the same shape used by the FAISS index

    This keeps the interface identical to the old "read from disk" version,
    so the rest of the codebase doesn't have to change.
    """
    all_docs: List[Dict] = []

    for ticker in tickers:
        t = ticker.upper().strip()
        print(f"[INFO] Fetching latest online 10-K for {t} ...")

        try:
            filing_url = _find_latest_10k_url(t)
            text = _extract_full_10k_text(filing_url)
        except Exception as e:
            print(f"[ERROR] Could not load 10-K for {t}: {e}")
            continue

        chunks = chunk_text_by_tokens(text)

        for i, chunk in enumerate(chunks):
            doc_id = f"{t}_10K_latest_chunk_{i}"
            all_docs.append(
                {
                    "id": doc_id,
                    "company": t,
                    "filing_type": "10K",
                    "year": "latest",
                    "quarter": None,
                    "chunk_id": i,
                    "text": chunk,
                    "source_file": f"{t}_10K_online",
                }
            )

    print(f"[INFO] Loaded {len(all_docs)} chunks from {len(tickers)} online filings")
    return all_docs



# Test
if __name__ == "__main__":
    docs = load_and_chunk_filings(["AAPL"])

    print("\nExample chunk:")
    if docs:
        print(docs[0]["text"][:500])
    else:
        print("No chunks returned.")
