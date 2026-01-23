from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class FileRef(BaseModel):
    filePath: Optional[str] = None

class MultimodalAnalysisRef(BaseModel):
    statId: Optional[str] = None
    filePath: Optional[str] = None


class ReportRequest(BaseModel):
    chargerStatus: Optional[FileRef] = None
    multimodalAnalysis: Optional[MultimodalAnalysisRef] = None
    openRequests: Optional[FileRef] = None
    requestOutbounds: Optional[FileRef] = None
    chargerStatusAnalysis: Optional[FileRef] = None


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


@router.post("/api/report", response_model=ReportResponse)
async def report(_: ReportRequest) -> ReportResponse:
    return ReportResponse(
        reportId=1,
        reportTitle="auto-generated report",
        createdTime=datetime.utcnow(),
        filePath="/reports/report_1.pdf",
        reportType="summary",
        prompt="auto",
    )
