import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_all
from src.hybrid_search import hybrid_search
from src.cross_reference import get_related_authorities, get_related_records
from src.llm_answer import generate_answer_stream
from src.config import CHROMA_DIR, BM25_DIR

st.set_page_config(
    page_title="법무부 기록관 검색",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def ensure_index():
    vec_exists = (CHROMA_DIR / "records_vectors.npz").exists()
    bm25_exists = (BM25_DIR / "records.pkl").exists()
    if not vec_exists or not bm25_exists:
        with st.spinner("인덱스 빌드 중... (최초 1회, 약 5분 소요)"):
            from scripts.build_index import main as build_main
            build_main()


ensure_index()


@st.cache_data
def load_data():
    return load_all()


data = load_data()
doc_map = {}
for docs in data.values():
    for d in docs:
        doc_map[d["id"]] = d

# --- Sidebar ---
with st.sidebar:
    st.title("법무부 기록관 검색")

    st.markdown("---")

    search_mode = st.radio(
        "검색 범위",
        ["통합", "기록물", "전거", "웹기록"],
        horizontal=True,
    )

    mode_to_collections = {
        "통합": ["records", "authorities", "websns"],
        "기록물": ["records"],
        "전거": ["authorities"],
        "웹기록": ["websns"],
    }

    st.markdown("---")
    st.markdown("**보유 현황**")
    col1, col2, col3 = st.columns(3)
    col1.metric("기록물", f"{len(data['records'])}")
    col2.metric("전거", f"{len(data['authorities'])}")
    col3.metric("웹기록", f"{len(data['websns']):,}")

    st.markdown("---")
    top_k = st.slider("검색 결과 수", 3, 20, 10)

    st.markdown("---")
    st.caption("법무부 기록관 mojarchives.go.kr")
    st.caption("Powered by Gemini + ChromaDB")


# --- Helper ---
def render_results(result_docs):
    for doc in result_docs:
        doc_type = doc.get("type", "")
        title = doc.get("title", "")
        meta = doc.get("metadata", {})

        if doc_type == "record":
            detail = doc.get("detail", {}).get("recordData", {})
            with st.expander(f"[기록물] {title}"):
                c1, c2 = st.columns(2)
                c1.write(f"**생산일자:** {meta.get('date', '')}")
                c2.write(f"**유형:** {meta.get('folder_type', '')}")
                st.write(f"**키워드:** {meta.get('keywords', '')}")
                st.write(f"**컬렉션:** {meta.get('collections', '')}")
                desc = detail.get("tleContentsText", "")
                if desc:
                    st.write(desc[:500])

                related = get_related_authorities(doc["uuid"])
                if related:
                    st.write("**관련 전거:**")
                    for a in related:
                        st.write(f"- {a['name']} ({a['type']})")

                st.link_button("원본 보기", f"https://mojarchives.go.kr/search/collectrecords/viewrecord/{doc['uuid']}")

        elif doc_type == "authority":
            detail = doc.get("detail", {}).get("authData", {})
            with st.expander(f"[전거] {title}"):
                st.write(f"**유형:** {meta.get('auth_type', '')} > {meta.get('auth_subtype', '')}")
                st.write(f"**기간:** {meta.get('start', '')} ~ {meta.get('end', '')}")
                if meta.get("alt_name"):
                    st.write(f"**이칭:** {meta['alt_name']}")
                memo = detail.get("tseAuthMemo", "")
                if memo:
                    st.write(memo[:500])

                related = get_related_records(doc["uuid"])
                if related:
                    st.write(f"**관련 기록물 ({len(related)}건):**")
                    for r in related[:5]:
                        st.write(f"- {r['title'][:50]}")

                st.link_button("원본 보기", f"https://mojarchives.go.kr/search/authority/viewauth/{doc['uuid']}")

        elif doc_type == "websns":
            with st.expander(f"[{meta.get('platform', 'SNS')}] {title[:80]}"):
                c1, c2 = st.columns(2)
                c1.write(f"**플랫폼:** {meta.get('platform', '')}")
                c2.write(f"**작성일:** {meta.get('date', '')}")
                st.write(f"**작성자:** {meta.get('writer', '')}")
                content = doc.get("full_content", "")
                if content:
                    st.write(content[:300])
                if meta.get("tags"):
                    st.write(f"**태그:** {meta['tags'][:200]}")
                if meta.get("url"):
                    st.link_button("원본 보기", meta["url"])


# --- Main Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "법무부 기록관의 기록물, 전거데이터, 웹기록을 검색할 수 있습니다. 무엇을 찾으시겠습니까?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "docs" in msg and msg["docs"]:
            st.markdown("---")
            st.write(f"**검색 결과 {len(msg['docs'])}건**")
            render_results(msg["docs"])

if prompt := st.chat_input("검색어를 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    collections = mode_to_collections[search_mode]

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            results = hybrid_search(prompt, collections=collections, top_k=top_k)

        result_docs = []
        for doc_id, score in results:
            doc = doc_map.get(doc_id)
            if doc:
                result_docs.append(doc)

        if result_docs:
            response = st.write_stream(generate_answer_stream(prompt, result_docs))

            st.markdown("---")
            st.write(f"**검색 결과 {len(result_docs)}건**")
            render_results(result_docs)
        else:
            response = "검색 결과가 없습니다. 다른 키워드로 시도해보세요."
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response, "docs": result_docs})
