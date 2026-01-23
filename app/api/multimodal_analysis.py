from datetime import datetime
from typing import Optional

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ImageInput(BaseModel):
    imgPath: Optional[str] = None


class SensorLog(BaseModel):
    sensorTime: Optional[datetime] = None
    totalChargingKwh: Optional[float] = None
    totalChargingMin: Optional[int] = None
    currentSoc: Optional[int] = None
    currentEnergyMeterValue: Optional[float] = None
    chargingv: Optional[float] = None
    charginga: Optional[float] = None
    outPower: Optional[float] = None
    chargingGunTemperature1: Optional[int] = None
    chargingGunTemperature2: Optional[int] = None
    types: Optional[int] = None

    class Config:
        extra = "allow"


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


@router.post("/api/multimodal_analysis", response_model=MultimodalAnalysisResponse)
async def multimodal_analysis(
    request: Request,
) -> MultimodalAnalysisResponse:
    try:
        raw_body = await request.body()
    except Exception:  # pragma: no cover - logging helper
        raw_body = b""
    logger.info("Multimodal request raw body: %s", raw_body.decode("utf-8", errors="replace"))
    try:
        parsed_payload = await request.json()
    except Exception:  # pragma: no cover - logging helper
        parsed_payload = None
    logger.info("Multimodal request parsed payload: %s", parsed_payload)
    return MultimodalAnalysisResponse(
        fireYN=False,
        fireDetails="no fire detected",
        brokenYN=False,
        brokeDetails="no damage detected",
        cleanYN=True,
        cleanDetails="appears clean",
    )
