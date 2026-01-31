# app/api/multimodal_analysis.py

from __future__ import annotations

from datetime import datetime
from typing import Optional, Any, Dict, List
import os
import logging
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# ✅ .env 로드 (app/.env)
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

logger = logging.getLogger(__name__)
router = APIRouter()


# ----------------------------
# 0) LOAD .env SAFELY
# ----------------------------
def _load_env_once() -> None:
    """
    app/.env 를 현재 파일 위치 기반으로 확실히 로드한다.
    - 이 파일: app/api/multimodal_analysis.py
    - 목표:   app/.env
    """
    if getattr(_load_env_once, "_done", False):
        return

    if load_dotenv is None:
        logger.warning("[ENV] python-dotenv not installed. Skip loading .env")
        _load_env_once._done = True
        return

    env_path = (Path(__file__).resolve().parents[1] / ".env")  # app/.env
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        logger.info("[ENV] loaded: %s", str(env_path))
    else:
        logger.warning("[ENV] not found: %s", str(env_path))

    _load_env_once._done = True


_load_env_once()


# ----------------------------
# 0) PATHS / SETTINGS
# ----------------------------
BASE_MODEL_DIR = Path(__file__).resolve().parent / "model"  # app/api/model

SENSOR_MODEL_PATH = os.getenv(
    "SENSOR_MODEL_PATH",
    str(BASE_MODEL_DIR / "lgbm_detail_label_drop_types.joblib"),
)
RESNET_MODEL_PATH = os.getenv(
    "RESNET_MODEL_PATH",
    str(BASE_MODEL_DIR / "multitask_final_best_v3.keras"),
)
YOLO_CLS_PATH = os.getenv(
    "YOLO_CLS_PATH",
    str(BASE_MODEL_DIR / "yolo_fine_tune_v2.pt"),
)

IMG_SIZE = 224
FIRE_CLASSES = ["fire", "smoke", "other"]
CLEAN_CLASSES = ["clean", "dirty"]

# Thresholds
T_YOLO = float(os.getenv("T_YOLO", "0.50"))
T_FIRE = float(os.getenv("T_FIRE", "0.40"))
T_SMOKE = float(os.getenv("T_SMOKE", "0.15"))
T_DIRTY = float(os.getenv("T_DIRTY", "0.80"))
T_SENSOR = float(os.getenv("T_SENSOR", "0.60"))

USE_LLM_NOTES = os.getenv("USE_LLM_NOTES", "1") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ✅ OpenAI client: lazy init (API KEY 없으면 서버 부팅되게)
_OPENAI_CLIENT: Optional[OpenAI] = None


def _get_openai_client() -> Optional[OpenAI]:
    """
    OPENAI_API_KEY 가 없으면 None.
    있으면 최초 1회만 OpenAI(api_key=...) 생성.
    """
    global _OPENAI_CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT


# ----------------------------
# 1) SCHEMAS
# ----------------------------
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


class Verdict(BaseModel):
    fireYN: bool = False
    dirtyYN: Optional[bool] = None  # 문제 있으면 True, fire 케이스는 None 가능
    faultYN: bool = False           # 센서 고장 위험
    notes: str = ""


class MultimodalAnalysisResponse(BaseModel):
    code: int = 200

    # 최종 룰 결과
    status: str = "OK"  # OK|ALERT|REJECT
    reasons: List[str] = Field(default_factory=list)

    verdict: Verdict = Field(default_factory=Verdict)

    # 하위호환 필드(기존)
    fireYN: bool = False
    brokenYN: bool = False
    cleanYN: bool = True

    # 디버깅/설명용
    details: Dict[str, Any] = Field(default_factory=dict)


# ----------------------------
# 2) UTIL
# ----------------------------
def ensure_real_image(img_path: str) -> str:
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    return str(p)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON found in model output:\n{text[:400]}")
    return json.loads(m.group(0))


# ----------------------------
# 3) PREPROCESS
# ----------------------------
def preprocess_for_keras(img_path: str) -> np.ndarray:
    ext = os.path.splitext(img_path)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
        img_bytes = tf.io.read_file(img_path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        return img.numpy()

    if ext == ".webp":
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
            arr = np.array(im, dtype=np.float32)
        img = tf.convert_to_tensor(arr)
        img = preprocess_input(img)
        img = tf.expand_dims(img, axis=0)
        return img.numpy()

    raise ValueError(f"Unsupported image extension: {ext}")


# ----------------------------
# 4) LOAD MODELS (요청 때 1회만 로드)
# ----------------------------
def _ensure_models_loaded(app) -> None:
    if getattr(app.state, "models_loaded", False):
        return

    logger.info("[LOAD] Sensor ML: %s", SENSOR_MODEL_PATH)
    app.state.sensor_model = joblib.load(SENSOR_MODEL_PATH)

    logger.info("[LOAD] ResNet multitask: %s", RESNET_MODEL_PATH)
    app.state.resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)

    logger.info("[LOAD] YOLO cls: %s", YOLO_CLS_PATH)
    app.state.yolo_model = YOLO(YOLO_CLS_PATH, task="classify")

    app.state.models_loaded = True
    logger.info("[LOAD] done")


# ----------------------------
# 5) RUNNERS
# ----------------------------
def run_resnet(resnet_model, img_path: str) -> Dict[str, Any]:
    x = preprocess_for_keras(img_path)
    pred = resnet_model.predict(x, verbose=0)

    if isinstance(pred, dict):
        fire_prob = np.asarray(pred["fire_output"][0], dtype=np.float32)
        clean_prob = np.asarray(pred["clean_output"][0], dtype=np.float32)
    else:
        out_names = [t.name.split("/")[0] for t in resnet_model.outputs]
        name_to_arr = {n: pred[i][0] for i, n in enumerate(out_names)}
        fire_prob = np.asarray(name_to_arr["fire_output"], dtype=np.float32)
        clean_prob = np.asarray(name_to_arr["clean_output"], dtype=np.float32)

    fire_map = dict(zip(FIRE_CLASSES, fire_prob.tolist()))
    clean_map = dict(zip(CLEAN_CLASSES, clean_prob.tolist()))

    return {
        "fire": {"probs": fire_map},
        "clean": {"probs": clean_map},
    }


def run_yolo_cls(yolo_model, img_path: str) -> Dict[str, Any]:
    r = yolo_model.predict(img_path, verbose=False)[0]
    probs = r.probs.data.cpu().numpy()

    names = yolo_model.names  # list or dict
    if isinstance(names, dict):
        inv_names = {v: k for k, v in names.items()}
        yes_idx = inv_names.get("yes", None)
        prob_map = {names[i]: float(probs[i]) for i in range(len(probs)) if i in names}
    else:
        prob_map = {names[i]: float(probs[i]) for i in range(len(probs))}
        yes_idx = names.index("yes") if "yes" in names else None

    score_yes = float(probs[yes_idx]) if yes_idx is not None else float(prob_map.get("yes", 0.0))
    pred_name = max(prob_map, key=prob_map.get) if prob_map else None

    return {"probs": prob_map, "pred": pred_name, "score_yes": score_yes}


def run_sensor_ml(sensor_model, sensor_row: dict) -> Dict[str, Any]:
    df = pd.DataFrame([sensor_row])
    feats = list(sensor_model.feature_names_in_)

    for c in feats:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feats]

    proba = sensor_model.predict_proba(X)[0]
    classes = list(sensor_model.classes_)
    prob_map = {str(int(classes[i])): float(proba[i]) for i in range(len(classes))}
    score_risk = float(prob_map.get("2", 0.0))

    return {"probs": prob_map, "pred": int(classes[np.argmax(proba)]), "score_risk": score_risk}


# ----------------------------
# 6) RULE ENGINE (최종 룰셋)
# ----------------------------
def apply_rules_for_prefinal(
    yolo_yes: float,
    resnet_out: Dict[str, Any],
    sensor_risk: float,
) -> Dict[str, Any]:
    fire_probs = resnet_out["fire"]["probs"]
    clean_probs = resnet_out["clean"]["probs"]

    fire_p = float(fire_probs.get("fire", 0.0))
    smoke_p = float(fire_probs.get("smoke", 0.0))
    dirty = float(clean_probs.get("dirty", 0.0))

    scores = {
        "yolo_yes": float(yolo_yes),
        "fire_p": fire_p,
        "smoke_p": smoke_p,
        "dirty_risk": dirty,
        "sensor_risk": float(sensor_risk),
        "overall": 0.0,
        "dirty_risk_eff": None,
        "T_YOLO": T_YOLO,
        "T_FIRE": T_FIRE,
        "T_SMOKE": T_SMOKE,
        "T_DIRTY": T_DIRTY,
        "T_SENSOR": T_SENSOR,
    }

    # (0) YOLO gate => REJECT + 고정 notes
    if yolo_yes < T_YOLO:
        status = "REJECT"
        reasons = ["no_target"]
        verdict_seed = {"fireYN": False, "dirtyYN": None, "faultYN": False}
        scores["overall"] = 0.0
        scores["dirty_risk_eff"] = None

        llm_review_fixed = {
            "status": "REJECT",
            "reasons": reasons,
            "verdict": {
                "fireYN": False,
                "dirtyYN": None,
                "faultYN": False,
                "notes": "충전기를 찾을 수 없습니다. 사업수행기관에게 CCTV 영상 교체 요청을 하십시오.",
            },
        }
        return {
            "status": status,
            "reasons": reasons,
            "scores": scores,
            "verdict_seed": verdict_seed,
            "resnet_mut": {"fire": resnet_out["fire"], "clean": resnet_out["clean"]},
            "llm_review_fixed": llm_review_fixed,
        }

    # (1) fire/smoke 판단 (fire 우선)
    is_fire = fire_p >= T_FIRE
    is_smoke = (not is_fire) and (smoke_p >= T_SMOKE)

    # (2) dirty 감쇠
    if is_fire:
        dirty_eff = dirty * 0.1
    elif is_smoke:
        dirty_eff = dirty * 0.5
    else:
        dirty_eff = dirty

    scores["dirty_risk_eff"] = float(dirty_eff)

    # (3) status 결정 (dirty OR sensor 단독도 ALERT)
    if is_fire:
        status = "ALERT"
        reasons = ["fire_detected"]
        resnet_mut = {"fire": resnet_out["fire"], "clean": None}
        scores["dirty_risk"] = None
    elif is_smoke:
        status = "ALERT"
        reasons = ["smoke_detected_possible_fire"]
        resnet_mut = {"fire": resnet_out["fire"], "clean": resnet_out["clean"]}
    else:
        reasons = []
        if dirty_eff >= T_DIRTY:
            reasons.append("dirty_detected")
        if sensor_risk >= T_SENSOR:
            reasons.append("sensor_risk")
        status = "ALERT" if reasons else "OK"
        resnet_mut = {"fire": resnet_out["fire"], "clean": resnet_out["clean"]}

    # (4) verdict_seed
    fireYN = bool(is_fire or is_smoke)
    faultYN = bool(sensor_risk >= T_SENSOR)
    dirtyYN = None if (resnet_mut["clean"] is None) else bool(dirty_eff >= T_DIRTY)
    verdict_seed = {"fireYN": fireYN, "dirtyYN": dirtyYN, "faultYN": faultYN}

    scores["overall"] = float(max(fire_p, smoke_p, dirty_eff, float(sensor_risk)))

    return {
        "status": status,
        "reasons": reasons,
        "scores": scores,
        "verdict_seed": verdict_seed,
        "resnet_mut": resnet_mut,
        "llm_review_fixed": None,
    }


# ----------------------------
# 7) LLM NOTES (notes only, 값은 강제 덮어쓰기)
# ----------------------------
def llm_notes_only(summary: dict) -> dict:
    c = _get_openai_client()
    if c is None:
        raise RuntimeError("OPENAI_API_KEY not set (check app/.env)")

    prompt = f"""
반드시 JSON만 출력(추가 텍스트/마크다운 금지).
출력 status는 INPUT.status 그대로.
출력 reasons는 INPUT.reasons 그대로(수정/번역/추가 금지).
출력 verdict.fireYN/dirtyYN/faultYN은 INPUT.verdict_seed 값을 그대로 복사(변경 금지).
너는 verdict.notes만 1~2문장 한국어로 유도리 있게 작성한다.

notes 작성 가이드:
- reasons에 fire_detected가 있으면: "화재 위험" 중심(연기 단정 X, 오염 언급 X)
- reasons에 smoke_detected_possible_fire가 있으면: "연기 징후로 화재 가능성" (화재 단정 금지)
- verdict_seed.faultYN이 true면: "고장/센서 이상 가능성" 언급 (false면 언급 금지)
- verdict_seed.dirtyYN이 true면: "오염/관리 필요" 언급
- dirtyYN이 null이면 오염/청결 관련 단정 금지.
- 점수로 강도 조절:
  * fire_p 또는 smoke_p가 매우 높으면(>=0.8) "매우 높음/즉시 확인 필요"
  * 중간(0.4~0.8) "주의/추가 확인 권장"
  * 낮으면 "낮음"

출력 JSON 스키마(키 고정):
{{
  "status": "OK|ALERT|REJECT",
  "reasons": ["..."],
  "verdict": {{
    "fireYN": true,
    "dirtyYN": true,
    "faultYN": false,
    "notes": "string"
  }}
}}

INPUT:
{json.dumps(summary, ensure_ascii=False)}
""".strip()

    resp = c.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _extract_json(resp.choices[0].message.content)


# ----------------------------
# 8) ROUTE
# ----------------------------
from fastapi import APIRouter, Request, HTTPException, Body

@router.post("/api/multimodal_analysis", response_model=MultimodalAnalysisResponse)
async def multimodal_analysis(
    payload: MultimodalAnalysisRequest = Body(...),
    request: Request = None,
) -> MultimodalAnalysisResponse:
    _ensure_models_loaded(request.app if request else None)

    # ✅ 기존 req 파싱 대신 payload 그대로 사용
    req = payload

    if not req.image or not req.image.imgPath:
        raise HTTPException(status_code=422, detail="image.imgPath is required")
    if not req.sensorLog:
        raise HTTPException(status_code=422, detail="sensorLog is required")

    try:
        img_path = ensure_real_image(req.image.imgPath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    sensor_row = req.sensorLog.model_dump()
    sensor_row.update({
        "total_charging_kwh": req.sensorLog.totalChargingKwh,
        "total_charging_min": req.sensorLog.totalChargingMin,
        "current_soc": req.sensorLog.currentSoc,
        "current_energy_meter_value": req.sensorLog.currentEnergyMeterValue,
        "out_power": req.sensorLog.outPower,
        "charging_gun_temperature1": req.sensorLog.chargingGunTemperature1,
        "charging_gun_temperature2": req.sensorLog.chargingGunTemperature2,
    })

    yolo_out = run_yolo_cls(request.app.state.yolo_model, img_path)
    resnet_out = run_resnet(request.app.state.resnet_model, img_path)
    sensor_out = run_sensor_ml(request.app.state.sensor_model, sensor_row)

    yolo_yes = float(yolo_out.get("score_yes", 0.0))
    sensor_risk = float(sensor_out.get("score_risk", 0.0))

    rule = apply_rules_for_prefinal(yolo_yes, resnet_out, sensor_risk)

    status = rule["status"]
    reasons = rule["reasons"]
    scores = rule["scores"]
    verdict_seed = rule["verdict_seed"]
    resnet_mut = rule["resnet_mut"]

    llm_review = None
    if status == "REJECT":
        llm_review = rule["llm_review_fixed"]
        verdict = llm_review["verdict"]
    else:
        verdict = {
            "fireYN": verdict_seed["fireYN"],
            "dirtyYN": verdict_seed["dirtyYN"],
            "faultYN": verdict_seed["faultYN"],
            "notes": "",
        }

        if USE_LLM_NOTES and os.getenv("OPENAI_API_KEY"):
            try:
                summary = {"status": status, "reasons": reasons, "scores": scores, "verdict_seed": verdict_seed}
                llm_out = llm_notes_only(summary)

                llm_out["status"] = status
                llm_out["reasons"] = reasons
                llm_out.setdefault("verdict", {})
                llm_out["verdict"]["fireYN"] = verdict_seed["fireYN"]
                llm_out["verdict"]["dirtyYN"] = verdict_seed["dirtyYN"]
                llm_out["verdict"]["faultYN"] = verdict_seed["faultYN"]
                llm_out["verdict"].setdefault("notes", "")

                llm_review = llm_out
                verdict = llm_out["verdict"] 
            except Exception as e:
                verdict["notes"] = f"LLM_ERROR: {e}"

    fireYN = bool(verdict_seed["fireYN"])
    brokenYN = bool(verdict_seed["faultYN"])
    dirtyYN = verdict_seed["dirtyYN"]
    cleanYN = True if dirtyYN is None else (not bool(dirtyYN))

    details = {
        "thresholds": {"T_YOLO": T_YOLO, "T_FIRE": T_FIRE, "T_SMOKE": T_SMOKE, "T_DIRTY": T_DIRTY, "T_SENSOR": T_SENSOR},
        "yolo": yolo_out,
        "resnet": resnet_mut,
        "sensor": sensor_out,
        "rule_scores": scores,
        "llm_review": llm_review,
    }

    return MultimodalAnalysisResponse(
        code=200,
        status=status,
        reasons=reasons,
        verdict=Verdict(**verdict),
        fireYN=fireYN,
        brokenYN=brokenYN,
        cleanYN=cleanYN,
        details=details,
    )

