from datetime import datetime
from typing import Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel

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


@router.post("/api/request_outbound", response_model=RequestOutboundResponse)
async def request_outbound(_: RequestOutboundRequest) -> RequestOutboundResponse:
    return RequestOutboundResponse(
        answer="Thank you for your report. We are reviewing the issue.",
        answerDt=datetime.utcnow(),
    )
