#!/usr/bin/env python3
"""검색 파이프라인 CLI 테스트"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_search import hybrid_search
from src.data_loader import load_all
from src.cross_reference import get_related_authorities, get_related_records


def main():
    print("데이터 로드 중...")
    data = load_all()
    doc_map = {}
    for docs in data.values():
        for d in docs:
            doc_map[d["id"]] = d

    queries = [
        "독립운동 관련 기록물",
        "감옥 규칙",
        "신규식",
        "출입국 여권",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"질문: {query}")
        print(f"{'='*60}")

        results = hybrid_search(query, top_k=5)

        for rank, (doc_id, score) in enumerate(results, 1):
            doc = doc_map.get(doc_id)
            if doc:
                print(f"\n  [{rank}] ({doc['type']}) {doc['title'][:60]}  (score: {score:.4f})")
                if doc["type"] == "record":
                    related = get_related_authorities(doc["uuid"])
                    if related:
                        print(f"      관련 전거: {', '.join(a['name'] for a in related)}")
                elif doc["type"] == "authority":
                    related = get_related_records(doc["uuid"])
                    if related:
                        print(f"      관련 기록물: {', '.join(r['title'][:30] for r in related[:3])}")


if __name__ == "__main__":
    main()
