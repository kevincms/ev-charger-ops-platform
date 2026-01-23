from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QnARequest(BaseModel):
    prompt: Optional[str] = None


class QnAResponse(BaseModel):
    #code: int = 200
    answer: Optional[str] = None


@router.post("/api/QnA", response_model=QnAResponse)
async def qna(_: QnARequest) -> QnAResponse:
    return QnAResponse(answer="해당 공문서는 2024년 전기차 충전 인프라 지원 사업의 대상, 지원 금액, 신청 절차를 안내하고 있습니다.")