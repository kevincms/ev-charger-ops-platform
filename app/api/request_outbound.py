'''
app/data/chargerLog 아래에서 YYYY-MM-DD 형식의 최신 날짜 폴더만 골라 그 안의 .jsonl만 읽도록 변경
request_outbound.py에 최신 날짜 폴더 선택 로직 추가 (_latest_date_dir, _iter_jsonl_files)
상태로그 스캔 대상이 최신 날짜 폴더로 제한됨
request_outbound에서 최신 상태로그 병합 유지
'''
import os, time, uuid, logging, json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Iterable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.openai_vectorstore_rag import rag_pipeline

router = APIRouter()
BASE_CHARGERLOG_DIR = Path(__file__).resolve().parents[1] / "data" / "chargerLog"


def _parse_yyyymmddhhmmss(value: str) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if len(s) != 14 or not s.isdigit():
        return None
    return datetime(
        int(s[0:4]), int(s[4:6]), int(s[6:8]),
        int(s[8:10]), int(s[10:12]), int(s[12:14]),
    )


def _latest_date_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates: List[Path] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if len(name) == 10 and name[4] == "-" and name[7] == "-":
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.name)


def _iter_jsonl_files(base: Path) -> Iterable[Path]:
    latest_dir = _latest_date_dir(base)
    if latest_dir is None:
        return []
    return latest_dir.glob("*.jsonl")


def _pick_time(row: Dict[str, Any]) -> Optional[datetime]:
    for k in ("statUpdDt", "lastTedt", "lastTsdt"):
        t = _parse_yyyymmddhhmmss(row.get(k, ""))
        if t:
            return t
    return None


def _normalize_chargerlog_row(row: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("lastTsdt", "lastTedt", "statUpdDt", "chgerTime"):
        v = row.get(k)
        t = _parse_yyyymmddhhmmss(v) if isinstance(v, str) else None
        if t:
            row[k] = t.isoformat()
    return row


def _load_latest_charger_status(stat_id: str, chger_id: str) -> Optional[Dict[str, Any]]:
    base = BASE_CHARGERLOG_DIR
    if not base.exists():
        return None

    latest_row = None
    latest_time = None

    for fp in _iter_jsonl_files(base):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    if row.get("statId") != stat_id or row.get("chgerId") != chger_id:
                        continue
                    t = _pick_time(row)
                    if t and (latest_time is None or t > latest_time):
                        latest_time = t
                        latest_row = _normalize_chargerlog_row(row)
        except Exception:
            continue

    return latest_row


class ChargerStatus(BaseModel):
    statId: Optional[str] = None
    zcodeDescription: Optional[str] = None
    zscodeDescription: Optional[str] = None
    busidDescription: Optional[str] = None
    statNm: Optional[str] = None
    addr: Optional[str] = None
    busiCall: Optional[str] = None
    year: Optional[int] = None
    chgerId: Optional[str] = None
    chgerType: Optional[str] = None
    output: Optional[str] = None
    method: Optional[str] = None
    chgerTime: Optional[datetime] = None
    lastTsdt: Optional[datetime] = None
    lastTedt: Optional[datetime] = None
    statUpdDt: Optional[datetime] = None
    stat: Optional[Union[str, int]] = None


class ChargerLogFileRequest(BaseModel):
    relPath: str
    maxRows: Optional[int] = None


class ChargerLogFileRow(BaseModel):
    line: int
    ok: bool
    error: Optional[str] = None
    data: Optional[ChargerStatus] = None


class ChargerLogFileResponse(BaseModel):
    code: int = 200
    total: int = 0
    ok: int = 0
    failed: int = 0
    rows: List[ChargerLogFileRow] = []


@router.post("/api/chargerlog/load", response_model=ChargerLogFileResponse)
async def load_chargerlog_file(payload: ChargerLogFileRequest) -> ChargerLogFileResponse:
    rel_path = (payload.relPath or "").strip()
    if not rel_path:
        raise HTTPException(status_code=422, detail="relPath is required.")

    base = BASE_CHARGERLOG_DIR.resolve()
    full = (base / rel_path).resolve()
    if base not in full.parents and base != full:
        raise HTTPException(status_code=400, detail="Invalid charger log path.")
    if not full.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {full}")

    rows: List[ChargerLogFileRow] = []
    total = ok = failed = 0
    max_rows = payload.maxRows

    with full.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if max_rows is not None and total >= int(max_rows):
                break
            total += 1
            raw = line.strip()
            if not raw:
                failed += 1
                rows.append(ChargerLogFileRow(line=idx, ok=False, error="Empty line"))
                continue
            try:
                data = json.loads(raw)
                data = _normalize_chargerlog_row(data if isinstance(data, dict) else {})
                cs = ChargerStatus(**data)
                ok += 1
                rows.append(ChargerLogFileRow(line=idx, ok=True, data=cs))
            except Exception as exc:
                failed += 1
                rows.append(ChargerLogFileRow(line=idx, ok=False, error=str(exc)))

    return ChargerLogFileResponse(
        total=total,
        ok=ok,
        failed=failed,
        rows=rows,
    )


class MultimodalAnalysisSummary(BaseModel):
    statId: Optional[str] = None
    zcodeDescription: Optional[str] = None
    zscodeDescription: Optional[str] = None
    busidDescription: Optional[str] = None
    statNm: Optional[str] = None
    addr: Optional[str] = None
    busiCall: Optional[str] = None
    year: Optional[int] = None
    chgerId: Optional[str] = None
    chgerType: Optional[str] = None
    output: Optional[str] = None
    method: Optional[str] = None
    sensorTime: Optional[datetime] = None
    multimodalId: Optional[int] = None
    fireYn: Optional[bool] = None
    fireDetails: Optional[str] = None
    brokeYn: Optional[bool] = None
    brokeDetails: Optional[str] = None
    cleanYn: Optional[bool] = None
    cleanDetails: Optional[str] = None
    imgsensoranalTime: Optional[datetime] = None


class RequestInfo(BaseModel):
    reqId: Optional[int] = None
    statId: Optional[str] = None
    chgerId: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    reqType: Optional[str] = None


class RequestOutboundRequest(BaseModel):
    chargerStatus: Optional[ChargerStatus] = None
    multimodalAnalysis: Optional[MultimodalAnalysisSummary] = None
    request: Optional[RequestInfo] = None


class RequestOutboundResponse(BaseModel):
    code: int = 200
    answer: Optional[str] = None
    answerDt: Optional[datetime] = None


def build_prompt(payload: RequestOutboundRequest) -> str:
    request = payload.request or RequestInfo()
    title = request.title or ""
    content = request.content or ""
    req_type = request.reqType or ""
    stat = payload.chargerStatus
    multimodal = payload.multimodalAnalysis
    status_text = ""
    status_missing_text = ""
    multimodal_text = ""
    if stat:
        stat_data = stat.dict(exclude_none=True)
        if stat_data:
            status_text = "chargerStatus: " + " ".join(f"{k}={v}" for k, v in stat_data.items())
            if "stat" in stat_data and stat_data.get("stat") is not None:
                status_missing_text = "chargerStatusStatMissing=false"
            else:
                status_missing_text = "chargerStatusStatMissing=true"
        else:
            status_text = "chargerStatus: (정보없음)"
            status_missing_text = "chargerStatusStatMissing=true"
    else:
        status_text = "chargerStatus: (정보없음)"
        status_missing_text = "chargerStatusStatMissing=true"
    if multimodal:
        multimodal_data = multimodal.dict(exclude_none=True)
        if multimodal_data:
            multimodal_text = "multimodalAnalysis: " + " ".join(
                f"{k}={v}" for k, v in multimodal_data.items()
            )
    req_type_text = f"reqType={req_type}" if req_type else ""
    return " ".join(
        part
        for part in [title, content, req_type_text, status_text, status_missing_text, multimodal_text]
        if part
    )


def normalize_answer(text: Optional[str]) -> str:
    return (text or "").strip()


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("OUTBOUND")


@router.post("/api/request_outbound", response_model=RequestOutboundResponse)
async def request_outbound(payload: RequestOutboundRequest) -> RequestOutboundResponse:
    trace = str(uuid.uuid4())[:8]
    t0 = time.time()

    req_id = payload.request.reqId if payload.request else None
    logger.info("[%s] START reqId=%s", trace, req_id)

    req_info = payload.request
    stat_id = req_info.statId if req_info else None
    chger_id = req_info.chgerId if req_info else None
    if stat_id and chger_id:
        latest = _load_latest_charger_status(stat_id, chger_id)
        if latest:
            payload.chargerStatus = ChargerStatus(**latest)

    api_key_loaded = bool(os.environ.get("OPENAI_API_KEY"))
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    logger.info("[%s] OPENAI_API_KEY loaded=%s", trace, api_key_loaded)
    logger.info("[%s] OPENAI_VECTOR_STORE_ID=%s", trace, vector_store_id)
    if not vector_store_id:
        logger.error("[%s] NO VECTOR STORE ID", trace)
        raise HTTPException(status_code=500, detail="OPENAI_VECTOR_STORE_ID is required.")

    prompt = build_prompt(payload)
    if not prompt:
        logger.warning("[%s] EMPTY PROMPT", trace)
        raise HTTPException(status_code=400, detail="Request content is empty.")

    logger.info("[%s] PROMPT len=%s preview=%r", trace, len(prompt), prompt[:500])

    try:
        answer = normalize_answer(rag_pipeline(prompt, vector_store_id=vector_store_id, trace=trace))
        if not answer:
            logger.warning("[%s] EMPTY ANSWER: retrying with stronger prompt", trace)
            retry_prompt = f"{prompt}\n\n답변을 반드시 한 문단으로 작성해 주세요."
            answer = normalize_answer(rag_pipeline(retry_prompt, vector_store_id=vector_store_id))
        if not answer:
            logger.error("[%s] STILL EMPTY ANSWER", trace)
            raise HTTPException(status_code=502, detail="AI returned an empty answer.")
        logger.info("[%s] ANSWER len=%s preview=%r", trace, len(answer), answer[:120])
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[%s] RAG FAILED: %s", trace, exc)
        raise HTTPException(status_code=500, detail=f"RAG failed: {type(exc).__name__}") from exc

    logger.info("[%s] END elapsed=%.2fs", trace, time.time() - t0)

    return RequestOutboundResponse(
        answer=answer,
        answerDt=datetime.utcnow(),
    )
