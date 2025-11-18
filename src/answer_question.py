from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from pathlib import Path

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VEC_DB_PATH = "chunk_db_faiss.index"
META_DB_PATH = "chunk_db_meta.json"

class ChunkDB:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = faiss.read_index(VEC_DB_PATH)
        self.meta = json.loads(Path(META_DB_PATH).read_text(encoding="utf-8"))

    def retrieve(self, query: str, k: int = 5):
        # embed query
        q_vec = self.model.encode([query], convert_to_numpy=True)
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10)
        q_vec = q_vec.astype(np.float32)

        # search
        distances, indices = self.index.search(q_vec, k)
        idxs = indices[0]
        sims = distances[0]

        results = []
        for rank, (i, score) in enumerate(zip(idxs, sims)):
            if i < 0:
                continue
            info = dict(self.meta[i])
            info["score"] = float(score)
            results.append(info)
        return results


if __name__ == "__main__":
    db = ChunkDB()

    question = "What is Ruby?"
    hits = db.retrieve(question, k=3)

    print("Top matches:")
    for h in hits:
        print("---")
        print(f"[{h['id']}] {h['title']} (pages {h['pages']}, score {h['score']:.3f})")
        print(h['preview'][:400], "...")
