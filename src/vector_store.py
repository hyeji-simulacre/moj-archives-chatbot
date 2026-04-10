"""ChromaDB 대신 numpy 기반 코사인 유사도 벡터 검색"""
import json
import numpy as np
from pathlib import Path
from src.config import CHROMA_DIR

INDEX_DIR = CHROMA_DIR  # 같은 경로 재사용


def _index_path(name: str):
    return INDEX_DIR / f"{name}_vectors.npz"


def _meta_path(name: str):
    return INDEX_DIR / f"{name}_meta.json"


def upsert_docs(collection_name: str, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    arr = np.array(embeddings, dtype=np.float32)
    # L2 정규화 (코사인 유사도용)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr = arr / norms

    np.savez_compressed(str(_index_path(collection_name)), embeddings=arr)

    meta = {"ids": ids, "metadatas": metadatas, "documents": documents}
    with open(_meta_path(collection_name), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    return len(ids)


def search(collection_name: str, query_embedding: list[float], n_results: int = 10, where: dict | None = None):
    idx_path = _index_path(collection_name)
    meta_path = _meta_path(collection_name)

    if not idx_path.exists() or not meta_path.exists():
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    data = np.load(str(idx_path))
    embeddings = data["embeddings"]

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # 쿼리 정규화
    q = np.array(query_embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) or 1)

    # 코사인 유사도 (내적, 이미 정규화됨)
    scores = embeddings @ q

    # where 필터 적용
    if where:
        mask = np.ones(len(scores), dtype=bool)
        for key, val in where.items():
            for i, m in enumerate(meta["metadatas"]):
                if m.get(key) != val:
                    mask[i] = False
        scores[~mask] = -1

    # top-k
    n = min(n_results, len(scores))
    top_idx = np.argsort(scores)[::-1][:n]

    result_ids = [meta["ids"][i] for i in top_idx if scores[i] > 0]
    result_distances = [float(1.0 - scores[i]) for i in top_idx if scores[i] > 0]
    result_metadatas = [meta["metadatas"][i] for i in top_idx if scores[i] > 0]
    result_documents = [meta["documents"][i] for i in top_idx if scores[i] > 0]

    return {
        "ids": [result_ids],
        "distances": [result_distances],
        "metadatas": [result_metadatas],
        "documents": [result_documents],
    }
