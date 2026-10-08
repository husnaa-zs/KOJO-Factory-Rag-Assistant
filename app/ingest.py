"""Reads documents, splits them into chunks, embeds them, and saves the index.

Run it with:  python -m app.ingest
"""
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from pypdf import PdfReader

from app import config
from app.embeddings import embed


def read_document(path: Path) -> list[tuple[Optional[int], str]]:
    """Return a list of (page_number, text). Text files have page None."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return []


def chunk_text(text: str, max_chars: int = 0, overlap: int = 0) -> list[str]:
    """Split text into chunks of roughly max_chars, breaking on paragraphs.
    Each new chunk starts with the tail of the previous one (the overlap),
    so an answer that sits on a boundary is not cut in half."""
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    overlap = overlap or config.CHUNK_OVERLAP

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail}\n\n{paragraph}"
        else:
            current = paragraph

        # A single very long paragraph: cut it into pieces
        while len(current) > max_chars * 1.5:
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap:]

    if current.strip():
        chunks.append(current)
    return chunks


def build_index() -> int:
    """Build the index from everything in data/docs. Returns the chunk count."""
    config.INDEX_DIR.mkdir(exist_ok=True)

    chunks: list[dict] = []
    for path in sorted(config.DOCS_DIR.rglob("*")):
        if not path.is_file():
            continue
        for page, text in read_document(path):
            for piece in chunk_text(text):
                chunks.append(
                    {"id": len(chunks), "source": path.name, "page": page, "text": piece}
                )

    if not chunks:
        raise RuntimeError(
            f"No documents found in {config.DOCS_DIR}. Add .txt, .md or .pdf files."
        )

    vectors = embed([c["text"] for c in chunks])
    np.save(config.INDEX_DIR / "vectors.npy", vectors)
    (config.INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2), encoding="utf-8"
    )
    return len(chunks)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks from {config.DOCS_DIR}")
