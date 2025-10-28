from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# LLM providers
from openai import OpenAI
import ollama as _ollama


def _load_cfg(path: str = "src/config.yaml"):
    return yaml.safe_load(Path(path).read_text())

def _load_store(cfg):
    index = faiss.read_index(cfg["index_path"])
    docs = np.load(cfg["store_path"], allow_pickle=True)
    meta = json.loads(Path(cfg["meta_path"]).read_text())
    return index, docs, meta


def _embed(q: str, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vec = model.encode([q], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")


def retrieve(query: str, k: int, cfg) -> List[Tuple[float, str, dict]]:
    index, docs, meta = _load_store(cfg)
    vec = _embed(query, cfg["embed_model"]) # (1, d)
    scores, idxs = index.search(vec, k)
    out: List[Tuple[float, str, dict]] = []
    for s, i in zip(scores[0], idxs[0]):
        text = docs[i].item() if hasattr(docs[i], 'item') else docs[i]
        m = meta[i]
        out.append((float(s), text, m))
    return out

def _format_context(topk: List[Tuple[float, str, dict]]) -> str:
    parts = []
    for score, text, m in topk:
        header = f"\n---\n# Symbol: {m['symbol']} (page {m['page']})\n"
        parts.append(header + text)
    return "\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    system_path = Path("prompts/system_prompt.txt")
    system = system_path.read_text().strip() if system_path.exists() else "You answer with the given context only."
    # The user said they'll fill the prompt—this is a minimal glue you can edit later.
    return (
    f"<SYSTEM>\n{system}\n</SYSTEM>\n\n"
    f"<CONTEXT>\n{context}\n</CONTEXT>\n\n"
    f"<QUESTION>\n{query}\n</QUESTION>\n"
    )

def _call_openai(prompt: str, model: str) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
    model=model,
    messages=[
    {"role": "system", "content": "Follow instructions in <SYSTEM> and use <CONTEXT>."},
    {"role": "user", "content": prompt},
    ],
    temperature=0.2,
    )
    return resp.choices[0].message.content


def _call_ollama(prompt: str, model: str) -> str:
    resp = _ollama.chat(model=model, messages=[
    {"role": "system", "content": "Follow instructions in <SYSTEM> and use <CONTEXT>."},
    {"role": "user", "content": prompt},
    ])
    return resp["message"]["content"]


def answer(query: str, cfg_path: str = "src/config.yaml") -> str:
    cfg = _load_cfg(cfg_path)
    k = int(cfg.get("k", 6))
    topk = retrieve(query, k, cfg)
    context = _format_context(topk)
    prompt = _build_prompt(query, context)

    provider = cfg.get("provider", "openai").lower()
    model = cfg.get("model")

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        return _call_openai(prompt, model)
    elif provider == "ollama":
    # OLLAMA_HOST can be set in env if remote
        return _call_ollama(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/qa.py \"Your question\" [config.yaml]")
        raise SystemExit(1)
    q = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else "src/config.yaml"
    print(answer(q, cfg))