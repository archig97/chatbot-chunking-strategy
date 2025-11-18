from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pathlib import Path
import json

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small+fast, good for semantic search
RAW_DB_PATH = "chunk_db_raw.json"
VEC_DB_PATH = "chunk_db_faiss.index"
META_DB_PATH = "chunk_db_meta.json"

def make_text_for_embedding(node):
    """
    Turn a node into a single string the embedder will see.
    """
    parts = [node["title"]]
    parts.extend(node["text_blocks"])
    return "\n\n".join(parts)

def build_vector_store():
    # load the AST chunks
    ast_nodes = json.loads(Path(RAW_DB_PATH).read_text(encoding="utf-8"))

    model = SentenceTransformer(EMBED_MODEL_NAME)

    corpus_texts = []
    meta = []
    for node in ast_nodes:
        text_for_vec = make_text_for_embedding(node)
        corpus_texts.append(text_for_vec)
        meta.append({
            "id": node["id"],
            "title": node["title"],
            "pages": node["pages"],
            # store a preview so we can show snippets in answers:
            "preview": text_for_vec[:500]
        })

    embeddings = model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    # build FAISS index (cosine similarity via Inner Product on normalized vectors)
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed_embeddings = embeddings / norms

    index = faiss.IndexFlatIP(dim)  # IP = inner product
    index.add(normed_embeddings.astype(np.float32))

    # save index and metadata
    faiss.write_index(index, VEC_DB_PATH)
    Path(META_DB_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Stored {len(meta)} chunks in FAISS with dim={dim}")

if __name__ == "__main__":
    build_vector_store()
