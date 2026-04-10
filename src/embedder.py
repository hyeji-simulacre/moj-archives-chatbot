import time
from google import genai
from src.config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    client = _get_client()
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config={"task_type": task_type},
            )
            all_embeddings.extend([e.values for e in result.embeddings])
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"  Rate limited at batch {i // EMBEDDING_BATCH_SIZE}, waiting 60s...")
                time.sleep(60)
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config={"task_type": task_type},
                )
                all_embeddings.extend([e.values for e in result.embeddings])
            else:
                raise
        if (i // EMBEDDING_BATCH_SIZE + 1) % 10 == 0:
            print(f"  Embedded {i + len(batch)}/{len(texts)}")
    return all_embeddings


def embed_query(text: str) -> list[float]:
    result = embed_texts([text], task_type="RETRIEVAL_QUERY")
    return result[0]
