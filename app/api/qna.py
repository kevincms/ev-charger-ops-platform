import json
import os
import threading
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from openai import OpenAI
from pydantic import BaseModel
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.graph import END, START, StateGraph

router = APIRouter()
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = 0.1

_client_rag: Optional[OpenAI] = None
_client_llm: Optional[OpenAI] = None
_vector_store_id: Optional[str] = None
_graph = None
_graph_lock = threading.Lock()

_history_store: Dict[str, InMemoryChatMessageHistory] = {}
_history_lock = threading.Lock()
_chain = None
_chain_lock = threading.Lock()


class QnARequest(BaseModel):
    prompt: Optional[str] = None
    session_id: Optional[str] = None # 대화 세션 식별자, 스레드 id를 사용할 경우 변수명 수정필요(처리하는 로직은 같음)


class QnAResponse(BaseModel):
    #code: int = 200
    answer: Optional[str] = None


class RAGState(TypedDict, total=False):
    question: str
    retrieval_results: List[Dict[str, Any]]
    evidence_text: str
    citations: List[Dict[str, Any]]
    draft: str
    verdict: str
    issues: List[str]
    final_answer: str
    chat_history: List[Dict[str, str]]
    errors: List[str]

# Colab Secrets에 아래 이름으로 저장해두는 걸 권장

def _ensure_clients() -> None:
    global _client_rag, _client_llm, _vector_store_id
    if _client_rag and _client_llm and _vector_store_id:
        return
    rag_key = os.environ.get("OPENAI_API_KEY")
    llm_key = os.environ.get("OPENAI_API_KEY")
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    if not rag_key or not llm_key or not vector_store_id:
        raise HTTPException(
            status_code=500,
            detail=(
                "Missing env vars: OPENAI_API_KEY, OPENAI_VECTOR_STORE_ID"
            ),
        )
    _client_rag = OpenAI(api_key=rag_key)
    _client_llm = OpenAI(api_key=llm_key)
    _vector_store_id = vector_store_id


def _format_evidence_snippets(
    results: List[Dict[str, Any]], max_chars_each: int = 900
) -> Tuple[str, List[Dict[str, Any]]]:
    if not results:
        return "", []

    lines: List[str] = []
    cites: List[Dict[str, Any]] = []
    for i, r in enumerate(results, start=1):
        score = r.get("score", None)
        text = r.get("text") or r.get("content") or ""
        text = (text[:max_chars_each] + "...") if len(text) > max_chars_each else text

        file_info = r.get("file", {}) or {}
        filename = file_info.get("filename") or file_info.get("name")
        file_id = file_info.get("id") or r.get("file_id")

        cites.append(
            {
                "rank": i,
                "score": score,
                "file_id": file_id,
                "filename": filename,
            }
        )

        header = f"[Doc {i}" + (
            f" | score={score:.3f}]" if isinstance(score, (int, float)) else "]"
        )
        if filename:
            header += f" ({filename})"
        lines.append(header + "\n" + (text.strip() if text else "(empty)"))

    return "\n\n".join(lines), cites


def _format_chat_history(history: List[Dict[str, str]], max_turns: int = 6) -> str:
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for h in recent:
        role = h.get("role", "user")
        content = (h.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _summarize_evidence_text(
    text: str, max_sentences: int = 2, max_chars: int = 240
) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(cleaned) if p.strip()]
    summary = " ".join(parts[:max_sentences]) if parts else cleaned
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip()
        if not summary.endswith((".", "!", "?")):
            summary += "..."
    return summary


def _build_evidence_summary(
    results: List[Dict[str, Any]], max_items: int = 2
) -> str:
    summaries: List[str] = []
    for r in results or []:
        text = r.get("text") or r.get("content") or ""
        s = _summarize_evidence_text(text, max_sentences=1)
        if s:
            summaries.append(s)
        if len(summaries) >= max_items:
            break
    if not summaries:
        return ""
    return "참고 요약: " + " ".join(summaries)


def _build_reference_block(
    results: List[Dict[str, Any]], citations: List[Dict[str, Any]]
) -> str:
    names: List[str] = []
    seen = set()
    missing_ids: List[str] = []
    has_results = bool(results)

    for c in citations or []:
        name = (c.get("filename") or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)

    def _pick_title(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        meta = r.get("metadata") or {}
        title = (
            meta.get("title")
            or meta.get("filename")
            or meta.get("file_name")
            or meta.get("name")
        )
        file_info = r.get("file", {}) or {}
        attrs = file_info.get("attributes") or file_info.get("metadata") or {}
        if not title:
            title = (
                attrs.get("title")
                or attrs.get("filename")
                or attrs.get("file_name")
                or attrs.get("name")
            )
        if not title:
            title = (
                file_info.get("filename")
                or file_info.get("name")
                or r.get("filename")
                or r.get("file_name")
                or r.get("name")
            )
        fid = file_info.get("id") or r.get("file_id") or r.get("id")
        return (str(title).strip() if title else None, fid)

    if not names:
        for r in results or []:
            title, fid = _pick_title(r)
            if title:
                if title not in seen:
                    names.append(title)
                    seen.add(title)
            elif fid:
                missing_ids.append(str(fid))

    if names:
        doc_line = "참고 문서: " + ", ".join(names)
    elif has_results and missing_ids:
        uniq: List[str] = []
        seen_ids = set()
        for fid in missing_ids:
            if fid not in seen_ids:
                uniq.append(str(fid))
                seen_ids.add(fid)
        sample = ", ".join(uniq[:3])
        suffix = f" (예: {sample})" if sample else ""
        doc_line = "참고 문서: (문서명 누락)" + suffix
    elif has_results:
        doc_line = "참고 문서: (문서명 누락)"
    else:
        doc_line = "참고 문서: (검색 결과 없음)"

    return doc_line


def _retrieval_node(state: RAGState) -> RAGState:
    _ensure_clients()
    q = state["question"]
    state.setdefault("errors", [])

    try:
        resp = _client_rag.responses.create(
            model=MODEL,
            input=q,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [_vector_store_id],
                    "max_num_results": 5,
                }
            ],
            include=["file_search_call.results"],
        )

        data = resp.model_dump()
        results = []
        for item in data.get("output", []):
            if item.get("type") == "file_search_call":
                results = item.get("results", []) or []
                break

        evidence_text, citations = _format_evidence_snippets(results)

        state["retrieval_results"] = results
        state["evidence_text"] = evidence_text
        state["citations"] = citations

    except Exception as e:
        state["errors"].append(f"retrieval failed: {repr(e)}")
        state["retrieval_results"] = []
        state["evidence_text"] = ""
        state["citations"] = []

    return state


def _draft_node(state: RAGState) -> RAGState:
    _ensure_clients()
    q = state["question"]
    evidence_text = state.get("evidence_text", "")
    history_text = _format_chat_history(state.get("chat_history", []))
    state.setdefault("errors", [])

    system = (
        "너는 환경공단 전기차 충전소 관제 시스템의 업무 문서 Q&A 어시스턴트다.\n"
        "반드시 '근거'에 기반해서만 답변하고, 근거에 없는 내용은 추측하지 말고 '문서 근거 부족'이라고 말해라.\n"
        "가능하면 체크리스트/절차 형태로 간결하게 작성해라.\n"
        "근거 번호(근거1/2 등)는 언급하지 말아라.\n"
        "사용자가 직전 질문/이전 대화 내용을 묻는 경우에는 문서 근거 대신 대화 히스토리로 답해도 된다.\n"
    )

    user = f"""[대화 히스토리]
{history_text if history_text else "(이전 대화 없음)"}

[질문]
{q}

[근거]
{evidence_text if evidence_text else "(검색된 근거 없음)"}
""".strip()

    try:
        resp = _client_llm.responses.create(
            model=MODEL,
            temperature=TEMPERATURE,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        state["draft"] = resp.output_text
    except Exception as e:
        state["errors"].append(f"draft failed: {repr(e)}")
        state["draft"] = "문서 근거 부족: (LLM 호출 실패)"

    return state


def _verify_node(state: RAGState) -> RAGState:
    _ensure_clients()
    q = state["question"]
    evidence_text = state.get("evidence_text", "")
    history_text = _format_chat_history(state.get("chat_history", []))
    draft = state.get("draft", "")
    state.setdefault("errors", [])

    system = (
        "너는 답변 검증(Verifier) 에이전트다.\n"
        "입력으로 주어진 '근거'와 '초안'을 비교해서, 초안의 모든 핵심 주장(claim)이 근거에 의해 뒷받침되는지 검사해라.\n"
        "규칙:\n"
        "1) 근거에 없는 내용은 삭제하거나 '문서 근거 부족'으로 표시해라.\n"
        "2) 사용자 질문에 직접 답하도록 구성해라.\n"
        "3) 최종 출력은 반드시 JSON 하나만 반환해라.\n"
        "근거 번호(근거1/2 등)는 언급하지 말아라.\n"
        "사용자가 직전 질문/이전 대화 내용을 묻는 경우에는 문서 근거 대신 대화 히스토리로 답해도 된다.\n"
        "JSON 스키마:\n"
        "{\n"
        '  "verdict": "PASS" | "FAIL",\n'
        '  "issues": ["문제 문장/이유", ...],\n'
        '  "final_answer": "검증 후 최종 답변(사용자에게 보여줄 텍스트)"\n'
        "}\n"
    )

    user = f"""[대화 히스토리]
{history_text if history_text else "(이전 대화 없음)"}

[질문]
{q}

[근거]
{evidence_text if evidence_text else "(검색된 근거 없음)"}

[초안]
{draft}
""".strip()

    try:
        resp = _client_llm.responses.create(
            model=MODEL,
            temperature=TEMPERATURE,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.output_text.strip()

        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            obj = (
                json.loads(text[start : end + 1])
                if (start != -1 and end != -1 and end > start)
                else {
                    "verdict": "FAIL",
                    "issues": ["Verifier JSON 파싱 실패"],
                    "final_answer": draft,
                }
            )

        state["verdict"] = obj.get("verdict", "FAIL")
        state["issues"] = obj.get("issues", [])
        state["final_answer"] = obj.get("final_answer", draft)

        if "문서 근거 부족" not in state["final_answer"]:
            summary_block = _build_evidence_summary(
                state.get("retrieval_results", []),
            )
            if summary_block:
                state["final_answer"] = (
                    state["final_answer"].rstrip() + "\n\n" + summary_block
                )
            reference_block = _build_reference_block(
                state.get("retrieval_results", []),
                state.get("citations", []),
            )
            if reference_block:
                state["final_answer"] = (
                    state["final_answer"].rstrip() + "\n\n" + reference_block
                )

    except Exception as e:
        state["errors"].append(f"verify failed: {repr(e)}")
        state["verdict"] = "FAIL"
        state["issues"] = ["Verifier 호출 실패"]
        state["final_answer"] = draft

        if "문서 근거 부족" not in state["final_answer"]:
            summary_block = _build_evidence_summary(
                state.get("retrieval_results", []),
            )
            if summary_block:
                state["final_answer"] = (
                    state["final_answer"].rstrip() + "\n\n" + summary_block
                )
            reference_block = _build_reference_block(
                state.get("retrieval_results", []),
                state.get("citations", []),
            )
            if reference_block:
                state["final_answer"] = (
                    state["final_answer"].rstrip() + "\n\n" + reference_block
                )

    return state


def _get_graph():
    global _graph
    if _graph is not None:
        return _graph
    with _graph_lock:
        if _graph is not None:
            return _graph
        graph = StateGraph(RAGState)
        graph.add_node("retrieval", _retrieval_node)
        graph.add_node("draft", _draft_node)
        graph.add_node("verify", _verify_node)
        graph.add_edge(START, "retrieval")
        graph.add_edge("retrieval", "draft")
        graph.add_edge("draft", "verify")
        graph.add_edge("verify", END)
        _graph = graph.compile()
        return _graph


def _get_history(session_id: str) -> InMemoryChatMessageHistory:
    with _history_lock:
        return _history_store.setdefault(session_id, InMemoryChatMessageHistory())


def _invoke_with_history(inputs: Dict[str, Any]) -> str:
    question = inputs.get("input", "")
    history = inputs.get("history", [])
    history_dicts = []
    for m in history:
        role = "assistant" if m.type == "ai" else "user"
        history_dicts.append({"role": role, "content": m.content})
    graph = _get_graph()
    out = graph.invoke({"question": question, "chat_history": history_dicts})
    return out.get("final_answer", "")


def _get_chain():
    global _chain
    if _chain is not None:
        return _chain
    with _chain_lock:
        if _chain is not None:
            return _chain
        base_chain = RunnableLambda(_invoke_with_history)
        _chain = RunnableWithMessageHistory(
            base_chain,
            _get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return _chain


@router.post("/api/QnA", response_model=QnAResponse)
async def qna(req: QnARequest) -> QnAResponse:
    prompt = (req.prompt or "").strip()
    if not prompt:
        return QnAResponse(answer="문서 근거 부족: (질문이 비어 있습니다)")

    if prompt in {"챗봇을 종료", "챗봇 종료", "종료", "quit", "exit"}: # 세션 종료 명령어
        with _history_lock:
            if req.session_id:
                _history_store.pop(req.session_id, None)
            else:
                _history_store.clear()
        return QnAResponse(answer="챗봇을 종료합니다.")

    chain = _get_chain()
    session_id = (req.session_id or "").strip() or "default"
    answer = await run_in_threadpool(
        chain.invoke,
        {"input": prompt},
        {"configurable": {"session_id": session_id}},
    )

    return QnAResponse(answer=answer)
