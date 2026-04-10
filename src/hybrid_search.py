from src.embedder import embed_query
from src.vector_store import search as vector_search
from src.bm25_index import search_bm25
from src.config import BM25_WEIGHT, SEMANTIC_WEIGHT, SEARCH_TOP_K


def _rrf_merge(bm25_results: list[tuple[str, float]], vector_results: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    scores = {}

    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + BM25_WEIGHT / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + SEMANTIC_WEIGHT / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query: str, collections: list[str] | None = None, where: dict | None = None, top_k: int | None = None) -> list[tuple[str, float]]:
    if top_k is None:
        top_k = SEARCH_TOP_K

    if collections is None:
        collections = ["records", "authorities", "websns"]

    query_embedding = embed_query(query)

    all_bm25 = []
    all_vector = []

    for coll in collections:
        bm25_res = search_bm25(coll, query, top_k=top_k * 2)
        all_bm25.extend(bm25_res)

        vec_res = vector_search(coll, query_embedding, n_results=top_k * 2, where=where)
        if vec_res and vec_res["ids"] and vec_res["ids"][0]:
            for doc_id, dist in zip(vec_res["ids"][0], vec_res["distances"][0]):
                score = 1.0 - dist
                all_vector.append((doc_id, score))

    merged = _rrf_merge(all_bm25, all_vector)
    return merged[:top_k]
