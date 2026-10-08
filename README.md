# Factory Knowledge RAG Assistant

Ask questions about factory documents (machine manuals, safety procedures,
quality rules) and get answers with sources.

Built with Python, FastAPI, Hugging Face embeddings (sentence-transformers)
and the Gemini API.

## How it works (RAG)

    Documents -> split into chunks -> embeddings (Hugging Face model, local)
                                              |
    Question -> embedding -> find the closest chunks (cosine similarity)
                                              |
                       closest chunks + question -> Gemini -> answer + sources

1. `app/ingest.py`    reads files in `data/docs`, splits them into chunks, embeds them
2. `app/retriever.py` embeds the question and finds the most similar chunks
3. `app/rag.py`       builds the prompt; if the best match is too weak it answers
                      "I could not find this" without calling the LLM
4. `app/llm.py`       calls Gemini (with retries when Google is busy)
5. `app/main.py`      FastAPI endpoints: `POST /ask`, `POST /reindex`, `GET /health`

The sample documents in `data/docs` are fictional. Replace them with your own
`.txt`, `.md` or `.pdf` files (do not use confidential company documents).

## Run it (Windows)

1. Install Python 3.10 or newer from https://www.python.org/downloads/
   During install, tick "Add python.exe to PATH". Check with: `python --version`
2. Open this folder in VS Code. Make sure you opened the folder that contains
   `requirements.txt` (the zip may create a folder inside a folder).
   Open a terminal: Terminal > New Terminal.
3. Create and activate a virtual environment:

       python -m venv venv
       venv\Scripts\activate

   Your prompt should now start with `(venv)`.
4. Install the packages (this takes a few minutes and downloads a few hundred MB):

       pip install -r requirements.txt

5. Create your key file. Get a free key at https://aistudio.google.com/apikey

       copy .env.example .env

   Open `.env` and paste your key after `GEMINI_API_KEY=` (no quotes, no spaces).
6. Build the index (the first run also downloads the embedding model, about 90 MB):

       python -m app.ingest

7. Start the server:

       uvicorn app.main:app --reload

8. Open http://127.0.0.1:8000 in your browser. API docs are at http://127.0.0.1:8000/docs

To stop the server press Ctrl+C. Next time, only repeat `venv\Scripts\activate`
and step 7.

## Evaluate retrieval

    python -m eval.run_eval

This checks that the right document is retrieved for each test question in
`eval/questions.json`, and that unrelated questions are rejected. It does not
call the LLM, so it is free. Add your own questions as you add documents.

## Adding your own documents

1. Put `.txt`, `.md` or `.pdf` files into `data/docs`.
2. Run `python -m app.ingest` again (or call `POST /reindex`).
3. Add a few questions to `eval/questions.json` and run the evaluation.

## Troubleshooting

- "running scripts is disabled" when activating the venv: run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt.
- "GEMINI_API_KEY is missing": `.env` is missing or misnamed; restart the server after editing it.
- "Gemini API error (404)": model name changed. Set `GEMINI_MODEL` in `.env`
  to a current model name from Google AI Studio.
- "Gemini API error (503)": Google is busy. Wait and retry, or try another model.
- "No module named app": run commands from the project folder (the one with `requirements.txt`).
- Good questions get "I could not find this": lower `MIN_SCORE` in `.env`
  (for example 0.2) and check with the evaluation script.

## Ideas to extend it

- Use a vector database (Chroma or FAISS) instead of a numpy file
- Add a reranker to improve which chunks are chosen
- Upload documents from the web page
- Streaming answers, conversation history, and a Docker setup
