"""
llm_client.py — Ollama wrapper for JobLens

Sends prompts to the locally-running 'joblens' Ollama model and
returns the generated text.  Falls back to any other Ollama model
if 'joblens' is unavailable (useful for testing without the fine-tuned GGUF).
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Iterator

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = "http://localhost:11434"
DEFAULT_MODEL    = "joblens"          # your fine-tuned model
FALLBACK_MODEL   = "llama3.1"         # used only if joblens is missing
TIMEOUT_SECONDS  = 120                # per-request timeout


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict:
    """Make a POST request to the Ollama API and return the parsed JSON."""
    url  = f"{OLLAMA_BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from exc


def _available_models() -> list[str]:
    """Return the list of model names currently pulled in Ollama."""
    url = f"{OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _resolve_model(preferred: str) -> str:
    """Use preferred model if available; otherwise fall back."""
    available = _available_models()
    # Check exact match or 'name:latest' match
    for m in available:
        if m == preferred or m.split(":")[0] == preferred:
            return m
    print(
        f"[llm_client] ⚠️  Model '{preferred}' not found. "
        f"Falling back to '{FALLBACK_MODEL}'."
    )
    return FALLBACK_MODEL


# ── Public API ─────────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = 0.3,
    num_predict: int   = 300,
    stream: bool       = False,
) -> str:
    """
    Send ``prompt`` to Ollama and return the generated text.

    Parameters
    ----------
    prompt : str
        The full prompt string (built by prompts.py).
    model : str
        Ollama model name (default: 'joblens').
    temperature : float
        Sampling temperature — lower = more deterministic.
    num_predict : int
        Max tokens to generate.
    stream : bool
        If True, stream tokens and print them as they arrive (CLI mode).

    Returns
    -------
    str
        The model's response text (stripped).
    """
    resolved = _resolve_model(model)

    payload = {
        "model":  resolved,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "stop": ["### Resume and Job:"],   # matches Modelfile stop token
        },
    }

    if stream:
        return _generate_streaming(payload)

    result = _post("/api/generate", payload)
    return result.get("response", "").strip()


def _generate_streaming(payload: dict) -> str:
    """Stream tokens to stdout and return the full assembled string."""
    url  = f"{OLLAMA_BASE_URL}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    full_response = []
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response.append(token)
                if chunk.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise ConnectionError(
            "Lost connection to Ollama while streaming."
        ) from exc
    print()  # newline after stream
    return "".join(full_response).strip()


def batch_generate(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
    **kwargs,
) -> list[str]:
    """
    Run generate() for each prompt in ``prompts`` and return all responses.

    Parameters
    ----------
    prompts : list[str]
        List of prompts (one per retrieved job).
    model : str
        Ollama model to use.
    **kwargs
        Forwarded to generate().

    Returns
    -------
    list[str]
        One response string per prompt, in the same order.
    """
    responses = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[llm_client] Generating response {i}/{len(prompts)} …")
        resp = generate(prompt, model=model, **kwargs)
        responses.append(resp)
    return responses


if __name__ == "__main__":
    # Quick smoke test
    test_prompt = (
        "Here is a candidate's resume:\n\n"
        "B.Tech CE student. Skills: Python, Flask, SQL.\n\n"
        "Here is a job posting:\n\n"
        "Title: Python Developer\nRequired Skills: Python, Flask, PostgreSQL\n\n"
        "Question: How well does this candidate match this job?\n"
        "Give a match score out of 100, list matching skills, missing skills, "
        "and one suggestion.\n\n"
        "Respond in this exact format:\n"
        "Match Score: <number>/100\n"
        "Matching Skills: <list>\n"
        "Missing Skills: <list>\n"
        "Suggestion: <sentence>"
    )

    print(f"[llm_client] Sending test prompt to model='{DEFAULT_MODEL}' …\n")
    response = generate(test_prompt, stream=True)
    print(f"\n[llm_client] ✅  Done.\nResponse:\n{response}")
