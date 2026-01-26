import os, time, uuid, logging
from datetime import datetime
from typing import Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.openai_vectorstore_rag import rag_pipeline

router = APIRouter()


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


#스프링 AiComplaintReq 구조에 맞춘 중첩 request
class RequestInfo(BaseModel):
    reqId: Optional[int] = None
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
    stat = payload.chargerStatus
    multimodal = payload.multimodalAnalysis
    status_text = ""
    multimodal_text = ""
    if stat:
        stat_data = stat.dict(exclude_none=True)
        if stat_data:
            status_text = "chargerStatus: " + " ".join(f"{k}={v}" for k, v in stat_data.items())
    if multimodal:
        multimodal_data = multimodal.dict(exclude_none=True)
        if multimodal_data:
            multimodal_text = "multimodalAnalysis: " + " ".join(
                f"{k}={v}" for k, v in multimodal_data.items()
            )
    return " ".join(part for part in [title, content, status_text, multimodal_text] if part)
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

    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    if not vector_store_id:
        logger.error("[%s] NO VECTOR STORE ID", trace)
        raise HTTPException(status_code=500, detail="OPENAI_VECTOR_STORE_ID is required.")

    prompt = build_prompt(payload)
    if not prompt:
        logger.warning("[%s] EMPTY PROMPT", trace)
        raise HTTPException(status_code=400, detail="Request content is empty.")

    logger.info("[%s] PROMPT len=%s preview=%r", trace, len(prompt), prompt[:120])

    try:
        answer = normalize_answer(rag_pipeline(prompt, vector_store_id=vector_store_id))
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
'''
@router.post("/api/request_outbound", response_model=RequestOutboundResponse)
async def request_outbound(payload: RequestOutboundRequest) -> RequestOutboundResponse:
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    if not vector_store_id:
        raise HTTPException(status_code=500, detail="OPENAI_VECTOR_STORE_ID is required.")
    prompt = build_prompt(payload)
    answer = rag_pipeline(prompt, vector_store_id=vector_store_id)
    return RequestOutboundResponse(
        answer=answer,
        answerDt=datetime.utcnow(),
    )


@router.post("/api/request_outbound", response_model=RequestOutboundResponse)
async def request_outbound(_: RequestOutboundRequest) -> RequestOutboundResponse:
    return RequestOutboundResponse(
        answer="Thank you for your report. We are reviewing the issue.",
        answerDt=datetime.utcnow(),
    )
'''
