import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.config import BM25_DIR

Path(BM25_DIR).mkdir(parents=True, exist_ok=True)

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            from kiwipiepy import Kiwi
            kiwi = Kiwi()

            def tokenize(text):
                tokens = kiwi.tokenize(text)
                return [t.form for t in tokens if len(t.form) > 1 or t.tag.startswith("N")]

            _tokenizer = tokenize
        except ImportError:
            def tokenize(text):
                text = re.sub(r"[^\w\s]", " ", text)
                return [w for w in text.split() if len(w) > 1]

            _tokenizer = tokenize
    return _tokenizer


def build_bm25(docs: list[dict], name: str) -> BM25Okapi:
    tokenize = _get_tokenizer()
    corpus = [tokenize(d["embed_text"]) for d in docs]
    doc_ids = [d["id"] for d in docs]

    bm25 = BM25Okapi(corpus)

    index_path = Path(BM25_DIR) / f"{name}.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids, "corpus": corpus}, f)

    return bm25


def load_bm25(name: str):
    index_path = Path(BM25_DIR) / f"{name}.pkl"
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["doc_ids"]


def search_bm25(name: str, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    bm25, doc_ids = load_bm25(name)
    tokenize = _get_tokenizer()
    tokens = tokenize(query)

    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
    return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]
