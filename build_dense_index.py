#!/usr/bin/env python3
"""Build a FAISS dense index from the corpus using sentence-transformers."""

import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "data", "cs288-sp26-a3-materials", "eecs_text_bs_rewritten.jsonl")
DOCS_PATH = os.path.join(SCRIPT_DIR, "corpus", "docs.pkl")
DENSE_INDEX_PATH = os.path.join(SCRIPT_DIR, "corpus", "dense_index.faiss")
DENSE_DOCS_PATH = os.path.join(SCRIPT_DIR, "corpus", "dense_docs.pkl")

MODEL_NAME = "google/embeddinggemma-300m"
BATCH_SIZE = 64


def main():
    # Load docs (already chunked by build_index.py)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    print(f"Loaded {len(docs)} chunks.")

    # Prepare texts for encoding - use title + text (truncated)
    texts = []
    for doc in docs:
        title = doc.get("title", "")
        text = doc["text"]
        # Truncate to ~256 words for embedding (model max is 256 tokens)
        words = text.split()[:200]
        combined = f"{title}. {' '.join(words)}" if title else ' '.join(words)
        texts.append(combined)

    # Encode
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index (Inner Product since embeddings are normalized = cosine sim)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # Save
    faiss.write_index(index, DENSE_INDEX_PATH)
    with open(DENSE_DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"Saved dense index to {DENSE_INDEX_PATH}")
    print(f"Saved dense docs to {DENSE_DOCS_PATH}")


if __name__ == "__main__":
    main()
