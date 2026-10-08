import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DOCS_DIR = BASE_DIR / "data" / "docs"
INDEX_DIR = BASE_DIR / "index"

# Hugging Face embedding model. Runs locally and is free.
# It is downloaded automatically the first time (about 90 MB).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Chunking
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))
# If the best matching chunk scores below this, we say "I don't know"
# instead of letting the LLM guess. Tune it using eval/run_eval.py.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))
