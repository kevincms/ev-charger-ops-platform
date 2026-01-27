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
    "충전기 고장": (
        ("고장", 3),
        ("오류", 2),
        ("작동안함", 3),
        ("작동 안", 3),
        ("작동불가", 3),
        ("충전불가", 3),
        ("충전 안", 2),
        ("충전 중단", 4),
        ("중단", 2),
        ("끊김", 2),
        ("멈춤", 2),
        ("먹통", 3),
        ("재시작", 2),
        ("다시 시작", 2),
        ("충전 시작 후", 2),
        ("충전 중", 1),
    ),
    "결제": (
        ("결제", 3),
        ("결제 실패", 4),
        ("결제수단", 3),
        ("결제 수단", 3),
        ("카드", 2),
        ("승인", 2),
        ("승인 실패", 3),
        ("요금", 2),
        ("과금", 2),
        ("청구", 2),
        ("환불", 2),
        ("등록", 1),
        ("앱", 1),
    ),
    "보조금": (
        ("보조금", 3),
        ("지원금", 3),
        ("보조", 2),
        ("혜택", 2),
        ("지원", 1),
    ),
    "기타": (),
}


def _score_category(text: str, keywords: tuple[tuple[str, int], ...]) -> int:
    score = 0
    for keyword, weight in keywords:
        if keyword.lower() in text:
            score += weight
    return score


def extract_and_classify(*, text: str) -> ComplaintResult:
    clean_text = " ".join(text.split()) if text else ""
    lowered = clean_text.lower()

    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "기타":
            continue
        scores[category] = _score_category(lowered, keywords)

    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            best = [c for c, s in scores.items() if s == max_score]
            if len(best) == 1:
                return ComplaintResult(clean_text=clean_text, category=best[0])
            # tie-breaker: prefer more specific categories over generic overlaps
            for preferred in ("결제", "충전기 고장", "보조금"):
                if preferred in best:
                    return ComplaintResult(clean_text=clean_text, category=preferred)

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
    keywords: list[str]
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


def _parse_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_json_from_text(text: str) -> dict | None:
    data = _parse_json_object(text)
    if isinstance(data, dict):
        return data
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return _parse_json_object(text[start : end + 1])


def _llm_classify(
    *,
    client: OpenAI,
    text: str,
    model: str = "gpt-4.1-mini",
) -> tuple[str | None, list[str]]:
    system = (
        "당신은 전기차 충전기 민원 분류 담당자입니다. "
        "민원 문장에서 핵심 키워드(3~6개)와 유형을 분류하세요. "
        "유형은 반드시 다음 중 하나여야 합니다: 충전기 고장, 결제, 보조금, 기타. "
        "출력은 JSON만 반환하세요."
        
    )
    user = f"""
[민원]
{text}

출력 JSON 형식:
{{
  "keywords": ["..."],
  "category": "충전기 고장|결제|보조금|기타"
}}
""".strip()

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = _extract_output_text(resp)
    if not raw:
        return None, []

    data = _extract_json_from_text(raw)
    if not isinstance(data, dict):
        return None, []

    category = data.get("category")
    if category not in ("충전기 고장", "결제", "보조금", "기타"):
        category = None

    keywords = data.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    return category, keywords


def classify_node(state: RAGState) -> RAGState:
    raw_text = state.get("raw_text", "")
    ctx = _get_ctx(state)
    client: OpenAI = ctx["client"]

    llm_category, llm_keywords = _llm_classify(client=client, text=raw_text)
    if llm_category:
        clean_text = " ".join(raw_text.split()) if raw_text else ""
        logger.info(
            "[RAG] classify llm clean_len=%s category=%s keywords=%s",
            len(clean_text),
            llm_category,
            ", ".join(llm_keywords) if llm_keywords else "none",
        )
        return {
            **state,
            "clean_text": clean_text,
            "category": llm_category,
            "keywords": llm_keywords,
        }

    result = extract_and_classify(text=raw_text)
    logger.info(
        "[RAG] classify clean_len=%s category=%s",
        len(result.clean_text),
        result.category,
    )
    return {
        **state,
        "clean_text": result.clean_text,
        "category": result.category,
    }


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

    system_prompt = (
        "민원 답변 초안을 작성하는 한국 환경 공단의 고객지원 담당자입니다. "
        "제공되지 않은 상태 정보는 임의로 추정하지 말고, 알 수 없다고 명시하세요. "
        "입력에 chargerStatusStatMissing=true가 있으면 운영상태(stat)를 추정하거나 언급하지 마세요."
    )
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
        "아래 초안을 검증하고 문제가 있으면 더 정확하고 안전하게 수정하세요. "
        "제공되지 않은 상태 정보는 임의로 추정하지 말고, 알 수 없다고 명시하세요. "
        "입력에 chargerStatusStatMissing=true가 있으면 운영상태(stat)를 추정하거나 언급하지 마세요."
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
