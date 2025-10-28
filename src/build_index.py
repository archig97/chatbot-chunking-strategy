from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import yaml

from extract_code import extract_pdf_code
from chunk_ast import chunk_blocks_python, Chunk



class Meta(BaseModel):
    id: str
    symbol: str
    page: int


def _load_blocks_json(path: str) -> List[Dict]:
    return json.loads(Path(path).read_text())


def build_index_from_pdf(pdf_path: str, config_path: str = "src/config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    index_path = Path(cfg["index_path"]).parent
    index_path.mkdir(parents=True, exist_ok=True)

    blocks_json = extract_pdf_code(pdf_path)
    blocks = _load_blocks_json(blocks_json)

    # AST chunking (Python-focused). For other languages, plug in another chunker.
    chunks: List[Chunk] = chunk_blocks_python(blocks)

    # Prepare corpus
    corpus_texts = [c.code for c in chunks]
    meta = [Meta(id=c.id, symbol=c.symbol, page=c.page).model_dump() for c in chunks]

    # Embeddings
    embed_model = SentenceTransformer(cfg["embed_model"])
    embs = embed_model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True)
    embs_np = np.asarray(embs, dtype="float32")

    # FAISS index
    dim = embs_np.shape[1]
    index = faiss.IndexFlatIP(dim) # inner-product with normalized vectors = cosine sim
    index.add(embs_np)

    faiss.write_index(index, cfg["index_path"])
    np.save(cfg["store_path"], np.array(corpus_texts, dtype=object))
    Path(cfg["meta_path"]).write_text(json.dumps(meta, indent=2))

    print(f"Indexed {len(chunks)} chunks from {pdf_path} → {cfg['index_path']}")


if __name__ == "__main__":
    '''
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/build_index.py <path/to.pdf> [config.yaml]")
        raise SystemExit(1)
    '''
    pdf = "data\pdfs\python_code.pdf"
    cfg = "src\config.yaml"
    build_index_from_pdf(pdf, cfg)