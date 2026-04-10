import chromadb
from src.config import CHROMA_DIR


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(name: str):
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_docs(collection_name: str, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]):
    collection = get_or_create_collection(collection_name)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            documents=documents[i:end],
        )
    return collection.count()


def search(collection_name: str, query_embedding: list[float], n_results: int = 10, where: dict | None = None):
    collection = get_or_create_collection(collection_name)
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return results
