"""
ingestion/embedder.py
Stage 1/3 — Generate text embeddings for chunks.

Uses BAAI/bge-base-en-v1.5 (768-dim, fast, good medical domain coverage).
Embeddings are stored in Qdrant during Stage 1 ingestion.
"""

from __future__ import annotations
import os
from typing import List, Dict
import numpy as np

# Lazy-load the model on first use
_model = None
MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading model: {MODEL_NAME} …")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embedder] Model loaded. Embedding dim: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of texts.

    Args:
        texts:         List of strings to embed
        batch_size:    Batch size for encoding
        show_progress: Show tqdm progress bar

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    model = _get_model()
    # BGE models benefit from a query prefix for queries (not documents)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # cosine similarity via dot product
        convert_to_numpy=True,
    )
    return embeddings


def embed_chunks(chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
    """
    Add embeddings to chunk dicts in-place.

    Args:
        chunks: Output of chunker.chunk_pages()

    Returns:
        Same chunks with "embedding" key added (list of floats)
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, batch_size=batch_size)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    print(f"[embedder] Embedded {len(chunks)} chunks (dim={embeddings.shape[1]})")
    return chunks


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string for retrieval.
    BGE models use a special query prefix for asymmetric retrieval.
    """
    model = _get_model()
    # BGE query prefix improves retrieval quality
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    emb = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return emb.tolist()


def get_embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()


if __name__ == "__main__":
    sample = ["Metformin treats Type 2 Diabetes.", "Hypertension causes headaches."]
    embs = embed_texts(sample)
    print(f"Shape: {embs.shape}")
    print(f"Sample vector (first 5 dims): {embs[0][:5]}")
