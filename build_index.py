#!/usr/bin/env python3
"""Build a BM25 index from the crawled corpus chunks."""

import json
import pickle
import re
import os
from rank_bm25 import BM25Okapi

CORPUS_PATH = "corpus/chunks.jsonl"
INDEX_PATH = "corpus/bm25_index.pkl"
DOCS_PATH = "corpus/docs.pkl"


def tokenize(text):
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'\w+', text.lower())


def main():
    # Load chunks
    chunks = []
    with open(CORPUS_PATH, "r") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} chunks.")

    # Deduplicate chunks by text hash
    seen = set()
    deduped = []
    for chunk in chunks:
        text_hash = hash(chunk["text"][:200])
        if text_hash not in seen:
            seen.add(text_hash)
            deduped.append(chunk)
    chunks = deduped
    print(f"After dedup: {len(chunks)} chunks.")

    # Tokenize - include URL path components for better matching
    corpus_tokens = []
    for chunk in chunks:
        # Extract meaningful tokens from URL path
        url_path = chunk.get("url", "").split("//")[-1]
        url_tokens = re.findall(r'[a-zA-Z]+', url_path)
        text = f"{chunk['title']} {' '.join(url_tokens)} {chunk['text']}"
        tokens = tokenize(text)
        corpus_tokens.append(tokens)

    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokens)

    # Save
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved BM25 index to {INDEX_PATH}")
    print(f"Saved docs to {DOCS_PATH}")


if __name__ == "__main__":
    main()
