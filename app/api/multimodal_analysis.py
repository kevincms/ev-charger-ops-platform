from datetime import datetime
from typing import Optional
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
from pydantic import BaseModel
from openai import OpenAI

logger = logging.getLogger(__name__)
router = APIRouter()

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

T_YOLO = float(os.getenv("T_YOLO", "0.50"))
T_SMOKE = float(os.getenv("T_SMOKE", "0.15"))
T_DIRTY = float(os.getenv("T_DIRTY", "0.80"))
T_SENSOR = float(os.getenv("T_SENSOR", "0.60"))

USE_LLM_FINAL = os.getenv("USE_LLM_FINAL", "1") == "1"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()

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


class MultimodalAnalysisResponse(BaseModel):
    code: int = 200
    fireYN: bool = False
    fireDetails: Optional[str] = None
    brokenYN: bool = False
    brokeDetails: Optional[str] = None
    cleanYN: bool = True
    cleanDetails: Optional[str] = None


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
        raise ValueError(f"No JSON found in model output:\n{text}")
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
def run_resnet(resnet_model, img_path: str):
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
        "fire": {"probs": fire_map, "score_risk": float(max(fire_map["fire"], fire_map["smoke"]))},
        "clean": {"probs": clean_map, "score_dirty": float(clean_map["dirty"])},
    }


def run_yolo_cls(yolo_model, img_path: str):
    r = yolo_model.predict(img_path, verbose=False)[0]
    probs = r.probs.data.cpu().numpy()
    names = yolo_model.names
    prob_map = {names[i]: float(probs[i]) for i in range(len(probs))}

    score_yes = float(
        prob_map.get(
            "yes",
            list(prob_map.values())[1] if len(prob_map) > 1 else 0.0,
        )
    )

    return {
        "probs": prob_map,
        "pred": max(prob_map, key=prob_map.get) if prob_map else None,
        "score_yes": score_yes,
    }


def run_sensor_ml(sensor_model, sensor_row: dict):
    df = pd.DataFrame([sensor_row])
    feats = list(sensor_model.feature_names_in_)

    for c in feats:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feats]

    proba = sensor_model.predict_proba(X)[0]
    classes = sensor_model.classes_
    prob_map = {str(int(classes[i])): float(proba[i]) for i in range(len(classes))}

    return {
        "probs": prob_map,
        "pred": int(classes[np.argmax(proba)]),
        "score_risk": float(prob_map.get("2", 0.0)),
    }


# ----------------------------
# 6) LLM FINAL
# ----------------------------
def llm_final_judge(summary: dict) -> dict:
    prompt = f"""
너는 전기차 충전기 위험 모니터링의 최종 판정자다.
반드시 JSON만 출력해라. 다른 텍스트/설명/마크다운 금지.
notes는 한국어로 출력하라.

규칙(중요):
- fireYN: 연기/불 의심이면 true
- cleanYN: 더러우면 false, 깨끗하면 true
- brokenYN: 고장/이상 의심이면 true
- reasons는 최대 3개

INPUT:
{json.dumps(summary, ensure_ascii=False)}

OUTPUT JSON:
{{
  "fireYN": true|false,
  "cleanYN": true|false,
  "brokenYN": true|false,
  "reasons": ["..."],
  "notes": "string"
}}
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You must output ONLY valid JSON. No extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _extract_json(resp.choices[0].message.content)


# ----------------------------
# 7) ROUTE (추론 + 응답매핑)
# ----------------------------
@router.post("/api/multimodal_analysis", response_model=MultimodalAnalysisResponse)
async def multimodal_analysis(request: Request) -> MultimodalAnalysisResponse:
    try:
        raw_body = await request.body()
    except Exception:
        raw_body = b""
    logger.info("Multimodal request raw body: %s", raw_body.decode("utf-8", errors="replace"))

    try:
        parsed_payload = await request.json()
    except Exception:
        parsed_payload = None
    logger.info("Multimodal request parsed payload: %s", parsed_payload)

    try:
        req = MultimodalAnalysisRequest.model_validate(parsed_payload or {})
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid payload: {e}")

    _ensure_models_loaded(request.app)

    if not req.image or not req.image.imgPath:
        raise HTTPException(status_code=422, detail="image.imgPath is required")
    if not req.sensorLog:
        raise HTTPException(status_code=422, detail="sensorLog is required")

    try:
        img_path = ensure_real_image(req.image.imgPath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    sensor_row = req.sensorLog.model_dump()
    sensor_row.update(
        {
            "total_charging_kwh": req.sensorLog.totalChargingKwh,
            "total_charging_min": req.sensorLog.totalChargingMin,
            "current_soc": req.sensorLog.currentSoc,
            "current_energy_meter_value": req.sensorLog.currentEnergyMeterValue,
            "out_power": req.sensorLog.outPower,
            "charging_gun_temperature1": req.sensorLog.chargingGunTemperature1,
            "charging_gun_temperature2": req.sensorLog.chargingGunTemperature2,
        }
    )

    yolo_out = run_yolo_cls(request.app.state.yolo_model, img_path)
    resnet_out = run_resnet(request.app.state.resnet_model, img_path)
    sensor_out = run_sensor_ml(request.app.state.sensor_model, sensor_row)

    yolo_yes = float(yolo_out.get("score_yes", 0.0))

    if yolo_yes < T_YOLO:
        return MultimodalAnalysisResponse(
            code=200,
            fireYN=False,
            fireDetails=f"REJECT(no_target_object): yolo_yes={yolo_yes:.3f} < T_YOLO={T_YOLO}",
            brokenYN=False,
            brokeDetails=f"REJECT(no_target_object): yolo_yes={yolo_yes:.3f} < T_YOLO={T_YOLO}",
            cleanYN=True,
            cleanDetails=f"REJECT(no_target_object): yolo_yes={yolo_yes:.3f} < T_YOLO={T_YOLO}",
        )

    fire_risk = float(resnet_out["fire"]["score_risk"])
    dirty = float(resnet_out["clean"]["score_dirty"])
    sensor_r = float(sensor_out.get("score_risk", 0.0))

    fireYN = fire_risk >= T_SMOKE
    cleanYN = not (dirty >= T_DIRTY)
    brokenYN = sensor_r >= T_SENSOR

    llm_notes = ""
    llm_reasons = []

    if USE_LLM_FINAL and os.getenv("OPENAI_API_KEY"):
        summary = {
            "imgPath": req.image.imgPath,
            "scores": {
                "yolo_yes": yolo_yes,
                "fire_risk": fire_risk,
                "dirty_risk": dirty,
                "sensor_risk": sensor_r,
                "T_YOLO": T_YOLO,
                "T_SMOKE": T_SMOKE,
                "T_DIRTY": T_DIRTY,
                "T_SENSOR": T_SENSOR,
            },
            "yolo_cls": yolo_out,
            "resnet": resnet_out,
            "sensor_ml": sensor_out,
        }
        try:
            llm_out = llm_final_judge(summary)
            fireYN = bool(llm_out.get("fireYN", fireYN))
            cleanYN = bool(llm_out.get("cleanYN", cleanYN))
            brokenYN = bool(llm_out.get("brokenYN", brokenYN))
            llm_notes = str(llm_out.get("notes", ""))
            llm_reasons = llm_out.get("reasons", []) or []
        except Exception as e:
            llm_notes = f"LLM_ERROR: {e}"
            llm_reasons = []

    return MultimodalAnalysisResponse(
        code=200,
        fireYN=fireYN,
        fireDetails=(
            f"fire_score(max(fire,smoke))={fire_risk:.3f}, "
            f"T_SMOKE={T_SMOKE} | fire_probs={resnet_out['fire']['probs']} | llm={llm_notes} {llm_reasons}"
        ),
        brokenYN=brokenYN,
        brokeDetails=(
            f"sensor_risk(class2)={sensor_r:.3f}, T_SENSOR={T_SENSOR} | "
            f"sensor_probs={sensor_out['probs']} | llm={llm_notes} {llm_reasons}"
        ),
        cleanYN=cleanYN,
        cleanDetails=(
            f"dirty_score={dirty:.3f}, T_DIRTY={T_DIRTY} | "
            f"clean_probs={resnet_out['clean']['probs']} | llm={llm_notes} {llm_reasons}"
        ),
    )
