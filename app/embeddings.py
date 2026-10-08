from functools import lru_cache

import numpy as np

from app import config


@lru_cache(maxsize=1)
def _model():
    # Imported here so the app starts quickly and the model loads only once.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Turn texts into vectors. Vectors are normalized (length 1), so the
    dot product of two vectors equals their cosine similarity."""
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")
