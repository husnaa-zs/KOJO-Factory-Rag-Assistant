import time

import requests

from app import config


def generate(prompt: str) -> str:
    """Call the Gemini API and return the text answer.
    Retries when Google is busy (503) or rate limiting us (429)."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env and restart the server.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_MODEL}:generateContent"
    )
    headers = {"Content-Type": "application/json", "x-goog-api-key": config.GEMINI_API_KEY}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }

    response = None
    for attempt in range(1, 6):
        response = requests.post(url, headers=headers, json=body, timeout=60)
        if response.status_code not in (429, 503):
            break
        if attempt < 5:
            time.sleep(attempt * 3)

    if response is None or not response.ok:
        status = response.status_code if response is not None else "no response"
        detail = response.text[:300] if response is not None else ""
        raise RuntimeError(f"Gemini API error ({status}): {detail}")

    data = response.json()
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return text
