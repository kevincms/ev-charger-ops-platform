from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

logger = logging.getLogger("RAG_STATE")


# -----------------------------
# 1) 민원 텍스트 추출/유형 분류
# -----------------------------
@dataclass(frozen=True)
class ComplaintResult:
    clean_text: str
    category: str


CATEGORY_KEYWORDS = {
    "충전기": ("충전기", "충전", "완속", "급속"),
    "고장": ("고장", "오류", "고장났다", "작동안함", "충전불가", "불량", "먹통"),
    "결제": ("결제", "요금", "과금", "청구", "환불", "카드"),
    "위치": ("위치", "주소", "길안내", "어디", "찾기"),
    "운영": ("운영", "운영시간", "오픈", "휴무"),
    "기타": (),
}


def extract_and_classify(*, text: str) -> ComplaintResult:
    clean_text = " ".join(text.split()) if text else ""
    lowered = clean_text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "기타":
            continue
        if any(keyword.lower() in lowered for keyword in keywords):
            return ComplaintResult(clean_text=clean_text, category=category)
    return ComplaintResult(clean_text=clean_text, category="기타")


# -----------------------------
# 2) 벡터 검색
# -----------------------------
def retrieve_references(
    client: OpenAI, *, vector_store_id: str, query: str, top_k: int = 3
) -> list[dict]:
    response = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=top_k,
    )
    results: list[dict] = []
    for item in response.data:
        content = getattr(item, "content", "")
        if not isinstance(content, str):
            content = str(content)

        metadata = getattr(item, "metadata", {}) or {}
        title = (
            metadata.get("title")
            or metadata.get("filename")
            or metadata.get("file_name")
            or metadata.get("name")
        )
        if not title:
            title = (
                getattr(item, "filename", None)
                or getattr(item, "file_name", None)
                or getattr(item, "name", None)
                or getattr(item, "file_id", None)
                or getattr(item, "id", None)
            )
        results.append({"title": title, "content": content})
    return results


# -----------------------------
# 3) LLM 출력 파싱
# -----------------------------
def _normalize(text: str | None) -> str:
    return (text or "").strip()


def _extract_output_text(resp: Any) -> str:
    txt = _normalize(getattr(resp, "output_text", None))
    if txt:
        return txt

    chunks: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                ctype = getattr(c, "type", None)
                if ctype in ("output_text", "text"):
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        chunks.append(t)
                    elif hasattr(t, "value"):
                        chunks.append(t.value)
                elif ctype == "refusal":
                    r = getattr(c, "refusal", None)
                    if isinstance(r, str):
                        chunks.append(r)
    return _normalize("\n".join(chunks))


def _reference_contents(references: list[dict]) -> list[str]:
    contents: list[str] = []
    for ref in references or []:
        if isinstance(ref, dict):
            text = ref.get("content")
            if isinstance(text, str) and text:
                contents.append(text)
        elif isinstance(ref, str):
            contents.append(ref)
    return contents


def _reference_titles(references: list[dict]) -> list[str]:
    titles: list[str] = []
    for ref in references or []:
        if isinstance(ref, dict):
            title = ref.get("title")
            if isinstance(title, str) and title.strip():
                titles.append(title.strip())
    return titles


def _unique_titles(titles: list[str]) -> list[str]:
    unique = []
    seen = set()
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# -----------------------------
# 4) LangGraph State
# -----------------------------
class RAGState(TypedDict, total=False):
    raw_text: str
    clean_text: str
    category: str
    references: list[dict]
    reference_titles: list[str]
    draft_answer: str
    final_answer: str
    error: str
    trace: str
    _ctx: dict


# -----------------------------
# 6) Node 구현 (분리된 함수)
# -----------------------------
def _get_ctx(state: RAGState) -> dict:
    return state.get("_ctx", {})


def classify_node(state: RAGState) -> RAGState:
    result = extract_and_classify(text=state.get("raw_text", ""))
    logger.info(
        "[RAG] classify clean_len=%s category=%s",
        len(result.clean_text),
        result.category,
    )
    return {**state, "clean_text": result.clean_text, "category": result.category}


def retrieve_node(state: RAGState) -> RAGState:
    clean_text = state.get("clean_text", "")
    if not clean_text:
        return {**state, "error": "empty_clean_text"}

    ctx = _get_ctx(state)
    client: OpenAI = ctx["client"]
    vector_store_id: str = ctx["vector_store_id"]
    top_k: int = ctx.get("top_k", 3)

    references = retrieve_references(
        client,
        vector_store_id=vector_store_id,
        query=f"{state.get('category', '')} {clean_text}".strip(),
        top_k=top_k,
    )
    logger.info("[RAG] retrieve refs=%s", len(references))
    if references:
        preview = references[0].get("content") if isinstance(references[0], dict) else references[0]
        if isinstance(preview, str):
            logger.info("[RAG] retrieve preview=%r", preview[:200])
    titles = _unique_titles(_reference_titles(references))
    if titles:
        logger.info("[RAG] retrieve titles=%s", ", ".join(titles))
    else:
        logger.info("[RAG] retrieve titles=none")
    return {**state, "references": references, "reference_titles": titles}


def draft_node(state: RAGState) -> RAGState:
    ctx = _get_ctx(state)
    client: OpenAI = ctx["client"]

    system_prompt = "민원 답변 초안을 작성하는 한국 환경 공단의 고객지원 담당자입니다."
    references_text = "\n\n---\n\n".join(_reference_contents(state.get("references", [])))

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "민원 유형/내용과 참고 문서를 바탕으로 답변 초안을 작성하세요.\n"
                    f"민원 유형: {state.get('category', '')}\n"
                    f"민원: {state.get('clean_text', '')}\n"
                    f"참고 문서:\n{references_text if references_text else '(참고 문서 없음)'}"
                ),
            },
        ],
    )

    draft_answer = _extract_output_text(resp)
    logger.info("[RAG] draft draft_len=%s", len(draft_answer))
    if not draft_answer:
        return {**state, "error": "empty_draft"}
    return {**state, "draft_answer": draft_answer}


def verify_node(state: RAGState) -> RAGState:
    ctx = _get_ctx(state)
    client: OpenAI = ctx["client"]

    draft_answer = state.get("draft_answer", "")
    if not draft_answer:
        return {**state, "error": "empty_draft"}

    category = state.get("category", "")
    complaint = state.get("clean_text", "")
    references_text = "\n\n---\n\n".join(_reference_contents(state.get("references", [])))

    system = (
        "당신은 한국 환경 공단의 전기차 충전기 고객지원 QA 담당자입니다. "
        "아래 초안을 검증하고 문제가 있으면 더 정확하고 안전하게 수정하세요."
    )

    user = f"""
[민원 유형]
{category}

[민원 내용]
{complaint}

[참고 문서/근거]
{references_text if references_text else "(참고 문서 없음)"}

[답변 초안]
{draft_answer}

요구사항:
1) 근거 없는 추정/판단은 제거하고, 근거가 있으면 근거와 맞는 표현 사용
2) 고객에게 필요한 다음 행동(안내/추가 정보 요청/센터 방문 안내 등) 명확히 제시
3) 3~5문장, 정중한 톤

출력은 JSON만:
{{
  "ok": true/false,
  "issues": ["..."],
  "revised_answer": "..."
}}
""".strip()

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = _extract_output_text(resp)
    if not raw:
        return {**state, "final_answer": draft_answer}

    try:
        data = json.loads(raw)
        ok = bool(data.get("ok", True))
        revised = _normalize(data.get("revised_answer", ""))
        if (not ok) and revised:
            final_answer = revised
        elif ok and revised:
            final_answer = revised
        else:
            final_answer = draft_answer
    except Exception:
        final_answer = draft_answer

    logger.info("[RAG] verify final_len=%s", len(final_answer))
    return {**state, "final_answer": final_answer}


def error_guard(state: RAGState) -> str:
    return "error" if state.get("error") else "ok"


# -----------------------------
# 7) Graph 구성
# -----------------------------
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "verify")

    graph.add_conditional_edges("retrieve", error_guard, {"ok": "draft", "error": END})
    graph.add_conditional_edges("draft", error_guard, {"ok": "verify", "error": END})
    graph.add_conditional_edges("verify", error_guard, {"ok": END, "error": END})

    return graph


# -----------------------------
# 7) Pipeline entry
# -----------------------------
def rag_pipeline(text: str, *, vector_store_id: str, trace: str | None = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)
    graph = build_graph()
    app = graph.compile()

    t0 = time.time()
    state = app.invoke(
        {
            "raw_text": text,
            "trace": trace,
            "_ctx": {
                "client": client,
                "vector_store_id": vector_store_id,
                "top_k": 3,
            },
        }
    )
    elapsed = time.time() - t0

    if state.get("error"):
        raise RuntimeError(state["error"])

    final_answer = state.get("final_answer") or state.get("draft_answer") or ""
    if trace:
        print(f"[RAG {trace}] final_len={len(final_answer)} elapsed={elapsed:.2f}s")
    return final_answer
