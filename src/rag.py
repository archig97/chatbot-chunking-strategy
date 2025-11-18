# textbook_rag.py
import os
import json
import math
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("TOP_K", "3"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.20"))


def load_json(path):
    """Load JSON from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine(u, v):
    """Cosine similarity between two vectors."""
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def embed(text, emb_model):
    """Get embedding from Ollama for the given text."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": emb_model,
        "prompt": text
    }
    resp = requests.post(url, json=payload)
    if not resp.ok:
        raise RuntimeError(f"Embedding failed: {resp.status_code} {resp.text}")
    data = resp.json()
    return data.get("embedding", [])


def top_k_by_cosine(index, query_vec, k, threshold):
    """Rank chunks by cosine similarity and return top K above threshold."""
    scored = []
    for chunk in index.get("chunks", []):
        score = cosine(query_vec, chunk.get("embedding", []))
        scored.append({**chunk, "score": score})

    scored.sort(key=lambda c: c["score"], reverse=True)
    filtered = [c for c in scored if c["score"] >= threshold]
    if filtered:
        return filtered[:k]
    return scored[:k]


def build_prompt(question, contexts):
    """Build prompt using textbook excerpts + question."""
    header = "\n".join([
        "You are a helpful teaching assistant answering questions about a textbook.",
        "You must answer ONLY using the provided textbook excerpts.",
        "If the answer is not clearly supported by the excerpts, reply exactly:",
        '"this is beyond my scope."',
        "",
        "Be concise, factual, and do not invent information."
    ])

    context_block_lines = []
    for i, c in enumerate(contexts, start=1):
        score = c.get("score", 0.0)
        text = c.get("text", "")
        context_block_lines.append(
            f"<<excerpt {i} (score={score:.2f})>>\n{text}"
        )

    context_block = "\n\n".join(context_block_lines)

    prompt_parts = [
        header,
        "",
        "--- TEXTBOOK EXCERPTS ---",
        context_block,
        "",
        "--- QUESTION ---",
        question,
        "",
        "--- ANSWER ---"
    ]
    return "\n".join(prompt_parts)


def call_ollama_generate(prompt):
    """Call Ollama /api/generate."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": GEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 220,
            "temperature": 0.1
        }
    }
    resp = requests.post(url, json=payload)
    if not resp.ok:
        raise RuntimeError(f"Generate failed: {resp.status_code} {resp.text}")
    data = resp.json()
    return (data.get("response") or "").strip()


def answer_question(question, index_path="data/index.json"):
    """
    Main RAG function:
    - Load textbook index (JSON with embModel + chunks)
    - Embed question
    - Retrieve top-k chunks
    - Ask Ollama to answer using those excerpts
    """
    try:
        index = load_json(index_path)
    except FileNotFoundError:
        return {"text": "this is beyond my scope."}

    chunks = index.get("chunks", [])
    if not chunks:
        return {"text": "this is beyond my scope."}

    emb_model = index.get("embModel")
    if not emb_model:
        # Fallback: let env config or default decide embedding model if needed
        return {"text": "this is beyond my scope."}

    # Embed question
    q_vec = embed(question, emb_model)

    # Retrieve top-k relevant chunks
    hits = top_k_by_cosine(index, q_vec, TOP_K, SIM_THRESHOLD) or []

    # Guardrail: low similarity → bail out
    if not hits or hits[0].get("score", 0.0) < SIM_THRESHOLD:
        return {"text": "this is beyond my scope."}

    # Build prompt from excerpts
    prompt = build_prompt(question or "", hits)

    try:
        raw_answer = call_ollama_generate(prompt)
    except Exception:
        return {"text": "this is beyond my scope."}

    # Simple safety / quality guard
    if not raw_answer:
        return {"text": "this is beyond my scope."}

    if any(
        raw_answer.lstrip().lower().startswith(prefix)
        for prefix in [
            "as an ai",
            "i don't have",
            "i do not have",
            "i'm not sure",
            "i am not sure"
        ]
    ):
        return {"text": "this is beyond my scope."}

    print(raw_answer)
    return {"text": raw_answer}
