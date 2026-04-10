from google import genai
from src.config import GEMINI_API_KEY, LLM_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def generate_answer(question: str, context_docs: list[dict]) -> str:
    context_parts = []
    for i, doc in enumerate(context_docs[:5], 1):
        doc_type = doc.get("type", "")
        title = doc.get("title", "")
        meta = doc.get("metadata", {})

        if doc_type == "record":
            detail = doc.get("detail", {}).get("recordData", {})
            desc = detail.get("tleContentsText", "")[:500] if detail.get("tleContentsText") else ""
            part = f"[기록물 {i}] {title}\n생산일자: {meta.get('date', '')}\n분류: {meta.get('folder_type', '')}\n키워드: {meta.get('keywords', '')}\n{desc}"
        elif doc_type == "authority":
            detail = doc.get("detail", {}).get("authData", {})
            memo = detail.get("tseAuthMemo", "")[:500] if detail.get("tseAuthMemo") else ""
            part = f"[전거 {i}] {title} ({meta.get('auth_type', '')})\n기간: {meta.get('start', '')} ~ {meta.get('end', '')}\n{memo}"
        elif doc_type == "websns":
            content = doc.get("full_content", "")[:300]
            part = f"[웹기록 {i}] {title}\n플랫폼: {meta.get('platform', '')}\n작성일: {meta.get('date', '')}\n{content}"
        else:
            part = f"[{i}] {title}"

        context_parts.append(part)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""당신은 법무부 기록관의 검색 도우미입니다.
아래 검색된 기록물 정보를 바탕으로 사용자의 질문에 한국어로 답변하세요.

규칙:
- 검색 결과에 있는 정보만 사용하세요
- 출처(기록물 제목, 생산일자)를 반드시 포함하세요
- 관련 기록물이 여러 건이면 요약 후 개별 소개하세요
- 검색 결과에 없는 내용은 '해당 정보가 검색 결과에 없습니다'라고 답하세요

검색 결과:
{context}

질문: {question}"""

    client = _get_client()
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text


def generate_answer_stream(question: str, context_docs: list[dict]):
    context_parts = []
    for i, doc in enumerate(context_docs[:5], 1):
        doc_type = doc.get("type", "")
        title = doc.get("title", "")
        meta = doc.get("metadata", {})

        if doc_type == "record":
            detail = doc.get("detail", {}).get("recordData", {})
            desc = detail.get("tleContentsText", "")[:500] if detail.get("tleContentsText") else ""
            part = f"[기록물 {i}] {title}\n생산일자: {meta.get('date', '')}\n분류: {meta.get('folder_type', '')}\n키워드: {meta.get('keywords', '')}\n{desc}"
        elif doc_type == "authority":
            detail = doc.get("detail", {}).get("authData", {})
            memo = detail.get("tseAuthMemo", "")[:500] if detail.get("tseAuthMemo") else ""
            part = f"[전거 {i}] {title} ({meta.get('auth_type', '')})\n기간: {meta.get('start', '')} ~ {meta.get('end', '')}\n{memo}"
        elif doc_type == "websns":
            content = doc.get("full_content", "")[:300]
            part = f"[웹기록 {i}] {title}\n플랫폼: {meta.get('platform', '')}\n작성일: {meta.get('date', '')}\n{content}"
        else:
            part = f"[{i}] {title}"

        context_parts.append(part)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""당신은 법무부 기록관의 검색 도우미입니다.
아래 검색된 기록물 정보를 바탕으로 사용자의 질문에 한국어로 답변하세요.

규칙:
- 검색 결과에 있는 정보만 사용하세요
- 출처(기록물 제목, 생산일자)를 반드시 포함하세요
- 관련 기록물이 여러 건이면 요약 후 개별 소개하세요
- 검색 결과에 없는 내용은 '해당 정보가 검색 결과에 없습니다'라고 답하세요

검색 결과:
{context}

질문: {question}"""

    client = _get_client()
    response = client.models.generate_content_stream(
        model=LLM_MODEL,
        contents=prompt,
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text
