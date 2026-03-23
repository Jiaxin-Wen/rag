#!/usr/bin/env python3
"""Build a BM25 index from the source corpus with paragraph-aware chunking."""

import json
import pickle
import re
import os
import hashlib
import unicodedata
from rank_bm25 import BM25Okapi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "data", "cs288-sp26-a3-materials", "eecs_text_bs_rewritten.jsonl")
INDEX_PATH = os.path.join(SCRIPT_DIR, "corpus", "bm25_index.pkl")
DOCS_PATH = os.path.join(SCRIPT_DIR, "corpus", "docs.pkl")

CHUNK_SIZE = 500  # words
OVERLAP = 75      # words


def strip_diacritics(text):
    """Remove diacritics for better matching (e.g. Gödel -> Godel)."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text):
    """Tokenizer: lowercase, strip diacritics, split on non-alphanumeric."""
    text = strip_diacritics(text.lower())
    return re.findall(r'\w+', text)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def extract_title(text):
    """Extract title from first line if it looks like a heading."""
    lines = text.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith('#'):
            return first_line.lstrip('#').strip()
        if len(first_line) < 200:
            return first_line
    return ''


def main():
    # Load source documents
    docs = []
    with open(CORPUS_PATH, "r") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents from source corpus.")

    # Chunk documents
    all_chunks = []
    for doc in docs:
        url = doc['url']
        text = doc['text']
        title = extract_title(text)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(url.encode()).hexdigest()[:12] + f'_{i}'
            all_chunks.append({
                'id': chunk_id,
                'url': url,
                'title': title,
                'text': chunk
            })
    print(f"Created {len(all_chunks)} chunks.")

    # Deduplicate by full text hash
    seen = set()
    deduped = []
    for chunk in all_chunks:
        text_hash = hash(chunk["text"])
        if text_hash not in seen:
            seen.add(text_hash)
            deduped.append(chunk)
    all_chunks = deduped
    print(f"After dedup: {len(all_chunks)} chunks.")

    # Tokenize - include title + URL path tokens + diacritic-stripped text
    corpus_tokens = []
    for chunk in all_chunks:
        url_path = chunk.get("url", "").split("//")[-1]
        url_tokens = re.findall(r'[a-zA-Z]+', url_path)
        text = f"{chunk['title']} {' '.join(url_tokens)} {chunk['text']}"
        tokens = tokenize(text)
        corpus_tokens.append(tokens)

    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokens)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved BM25 index to {INDEX_PATH}")
    print(f"Saved {len(all_chunks)} docs to {DOCS_PATH}")


if __name__ == "__main__":
    main()
