from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="AI Response Server")


class ImageInput(BaseModel):
    imgPath: Optional[str] = None


class SensorLog(BaseModel):
    sensorTime: Optional[datetime] = None
    statUpdDt: Optional[str] = None
    note: Optional[str] = None
    acInputVoltageV: Optional[float] = None
    acFrequencyHz: Optional[float] = None
    currentA: Optional[float] = None
    powerKw: Optional[float] = None
    energyKwh: Optional[float] = None
    acLeakageMa: Optional[float] = None
    groundOk: Optional[bool] = None
    cabinetTempC: Optional[float] = None
    ambientTempC: Optional[float] = None
    humidityPct: Optional[float] = None
    connectorT: Optional[float] = None
    ambientT: Optional[float] = None
    batteryT: Optional[float] = None


class MultimodalAnalysisRequest(BaseModel):
    image: Optional[ImageInput] = None
    sensorLog: Optional[SensorLog] = None


class MultimodalAnalysisResponse(BaseModel):
    code: int = 200
    fireYN: bool = False
    fireDetails: Optional[str] = None
    brokenYN: bool = False
    brokeDetails: Optional[str] = None
    cleanYN: bool = True
    cleanDetails: Optional[str] = None


class FileRef(BaseModel):
    file_path: Optional[str] = None


class MultimodalAnalysisRef(BaseModel):
    statId: Optional[int] = None
    file_path: Optional[str] = None


class ReportRequest(BaseModel):
    chargerStatus: Optional[FileRef] = None
    multimodalAnalysis: Optional[MultimodalAnalysisRef] = None
    openRequests: Optional[FileRef] = None
    requestOutbounds: Optional[FileRef] = None
    chargerStatusAnalysis: Optional[dict] = Field(default=None)


class ReportResponse(BaseModel):
    code: int = 200
    reportId: Optional[int] = None
    reportTitle: Optional[str] = None
    createdTime: Optional[datetime] = None
    filePath: Optional[str] = None
    reportType: Optional[str] = None
    prompt: Optional[str] = None
    dataStartTime: Optional[datetime] = None
    dataEndTime: Optional[datetime] = None


class ChargerStatus(BaseModel):
    statId: Optional[int] = None
    zcodeDescription: Optional[str] = None
    zscodeDescription: Optional[str] = None
    busidDescription: Optional[str] = None
    statNm: Optional[str] = None
    addr: Optional[str] = None
    busiCall: Optional[str] = None
    year: Optional[str] = None
    chgerId: Optional[int] = None
    chgerType: Optional[int] = None
    output: Optional[str] = None
    method: Optional[str] = None
    chgerTime: Optional[datetime] = None
    lastTsdt: Optional[datetime] = None
    lastTedt: Optional[datetime] = None
    statUpdDt: Optional[datetime] = None
    stat: Optional[str] = None


class MultimodalAnalysisSummary(BaseModel):
    statId: Optional[int] = None
    zcodeDescription: Optional[str] = None
    zscodeDescription: Optional[str] = None
    busidDescription: Optional[str] = None
    statNm: Optional[str] = None
    addr: Optional[str] = None
    busiCall: Optional[str] = None
    year: Optional[str] = None
    chgerId: Optional[int] = None
    chgerType: Optional[int] = None
    output: Optional[str] = None
    method: Optional[str] = None
    sensorTime: Optional[datetime] = None
    multimodalId: Optional[int] = None
    fireYN: Optional[bool] = None
    fireDetails: Optional[str] = None
    brokeYN: Optional[bool] = None
    brokeDetails: Optional[str] = None
    cleanYN: Optional[bool] = None
    cleanDetails: Optional[str] = None
    imgsensoranalTime: Optional[datetime] = None


class RequestOutboundRequest(BaseModel):
    chargerStatus: Optional[ChargerStatus] = None
    multimodalAnalysis: Optional[MultimodalAnalysisSummary] = None
    reqId: Optional[int] = None
    title: Optional[str] = None
    content: Optional[str] = None
    reqType: Optional[str] = None


class RequestOutboundResponse(BaseModel):
    code: int = 200
    answer: Optional[str] = None
    answerDt: Optional[datetime] = None


class QnARequest(BaseModel):
    prompt: Optional[str] = None


class QnAResponse(BaseModel):
    code: int = 200
    answer: Optional[str] = None


@app.post("/api/report/multimodal_analysis", response_model=MultimodalAnalysisResponse)
async def multimodal_analysis(_: MultimodalAnalysisRequest) -> MultimodalAnalysisResponse:
    return MultimodalAnalysisResponse(
        fireYN=False,
        fireDetails="no fire detected",
        brokenYN=False,
        brokeDetails="no damage detected",
        cleanYN=True,
        cleanDetails="appears clean",
    )


@app.post("/api/report", response_model=ReportResponse)
async def report(_: ReportRequest) -> ReportResponse:
    return ReportResponse(
        reportId=1,
        reportTitle="auto-generated report",
        createdTime=datetime.utcnow(),
        filePath="/reports/report_1.pdf",
        reportType="summary",
        prompt="auto",
    )


@app.post("/api/request_outbound", response_model=RequestOutboundResponse)
async def request_outbound(_: RequestOutboundRequest) -> RequestOutboundResponse:
    return RequestOutboundResponse(
        answer="Thank you for your report. We are reviewing the issue.",
        answerDt=datetime.utcnow(),
    )


@app.post("/api/QnA", response_model=QnAResponse)
async def qna(_: QnARequest) -> QnAResponse:
    return QnAResponse(answer="This is a placeholder answer.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)

# http://localhost:8888/docs
# http://localhost:8888/redoc