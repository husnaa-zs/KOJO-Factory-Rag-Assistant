from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.rag import answer_question
from app.retriever import reindex

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Factory Knowledge Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


class Source(BaseModel):
    source: str
    page: Optional[int] = None
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    answered: bool
    sources: list[Source]


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": config.GEMINI_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        return answer_question(request.question.strip(), request.top_k)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error))


@app.post("/reindex")
def reindex_documents():
    """Rebuild the index after you add or change files in data/docs."""
    try:
        return {"chunks_indexed": reindex()}
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))
