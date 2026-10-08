from app import config
from app.llm import generate
from app.retriever import search

NOT_FOUND = (
    "I could not find this in the factory documents, so I won't guess. "
    "Try rephrasing the question, or check with your supervisor."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = f", page {hit['page']}" if hit["page"] else ""
        blocks.append(f"[{i}] (source: {hit['source']}{page})\n{hit['text']}")
    context = "\n\n".join(blocks)

    return f"""You are a factory knowledge assistant. Answer the question using ONLY the context below.

Rules:
- If the context does not contain the answer, reply exactly: "I could not find this in the factory documents."
- Do not use outside knowledge and do not guess.
- Be short and clear. Use numbered steps for procedures.
- After each fact, cite the source number like [1] or [2].
- For safety procedures, never skip or reorder steps.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def answer_question(question: str, top_k: int) -> dict:
    hits = search(question, top_k)

    # Retrieval guard: if even the best chunk is a weak match,
    # skip the LLM entirely instead of risking a made-up answer.
    if not hits or hits[0]["score"] < config.MIN_SCORE:
        return {"answer": NOT_FOUND, "answered": False, "sources": hits}

    answer = generate(build_prompt(question, hits))
    return {"answer": answer, "answered": True, "sources": hits}
