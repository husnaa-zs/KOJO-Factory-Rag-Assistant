import json
from typing import Optional

import numpy as np

from app import config
from app.embeddings import embed
from app.ingest import build_index

_vectors: Optional[np.ndarray] = None
_chunks: Optional[list[dict]] = None


def _files_exist() -> bool:
    return (config.INDEX_DIR / "vectors.npy").exists() and (
        config.INDEX_DIR / "chunks.json"
    ).exists()


def _load() -> None:
    global _vectors, _chunks
    _vectors = np.load(config.INDEX_DIR / "vectors.npy")
    _chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))


def _ensure_loaded() -> None:
    if not _files_exist():
        build_index()
    if _vectors is None or _chunks is None:
        _load()


def reindex() -> int:
    """Rebuild the index from data/docs (use after adding documents)."""
    count = build_index()
    _load()
    return count


def search(query: str, top_k: int = 0) -> list[dict]:
    """Return the top_k chunks most similar to the query, best first."""
    _ensure_loaded()
    top_k = top_k or config.TOP_K
    query_vector = embed([query])[0]
    scores = _vectors @ query_vector  # cosine similarity (vectors are normalized)
    best = np.argsort(-scores)[:top_k]
    return [{**_chunks[i], "score": round(float(scores[i]), 3)} for i in best]
