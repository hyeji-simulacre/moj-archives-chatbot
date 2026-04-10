#!/usr/bin/env python3
"""법무부 기록관 검색 인덱스 빌드 (ChromaDB + BM25)"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all
from src.embedder import embed_texts
from src.vector_store import upsert_docs, get_or_create_collection
from src.bm25_index import build_bm25


def build_collection(name: str, docs: list[dict]):
    print(f"\n=== {name} ({len(docs)}건) ===")

    # BM25 인덱스
    print(f"  BM25 인덱스 빌드...")
    build_bm25(docs, name)
    print(f"  BM25 완료")

    # 임베딩
    print(f"  임베딩 생성 중...")
    texts = [d["embed_text"] for d in docs]
    start = time.time()
    embeddings = embed_texts(texts)
    elapsed = time.time() - start
    print(f"  임베딩 완료 ({elapsed:.1f}s)")

    # ChromaDB 저장
    print(f"  ChromaDB 저장 중...")
    ids = [d["id"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    count = upsert_docs(name, ids, embeddings, metadatas, texts)
    print(f"  ChromaDB 저장 완료 ({count}건)")


def main():
    print("법무부 기록관 검색 인덱스 빌드 시작")
    total_start = time.time()

    data = load_all()

    for name, docs in data.items():
        build_collection(name, docs)

    total_elapsed = time.time() - total_start
    print(f"\n전체 완료 ({total_elapsed:.1f}s)")
    print(f"  기록물: {len(data['records'])}건")
    print(f"  전거: {len(data['authorities'])}건")
    print(f"  웹기록: {len(data['websns'])}건")


if __name__ == "__main__":
    main()
