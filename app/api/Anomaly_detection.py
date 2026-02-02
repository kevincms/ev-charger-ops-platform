# app/api/Anomaly_detection.py
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/anomaly", tags=["anomaly"])

# ====== status sets (training logic) ======
NORMAL = {2, 3}
BAD_FOR_FEATURE = {1, 4, 5, 9}  # 9 = 상태미확인 포함

# ====== bundle path ======
DEFAULT_BUNDLE_PATH = r"D:\FastAPI\ev-charger-ops-platform\app\api\model\lgb_risk_model_7days_bundle.joblib"


@lru_cache(maxsize=1)
def _load_bundle() -> Dict[str, Any]:
    path = os.getenv("EV_MODEL_BUNDLE_PATH", DEFAULT_BUNDLE_PATH)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model bundle not found: {path}\n"
            f"Set env EV_MODEL_BUNDLE_PATH to your joblib bundle path."
        )
    b = joblib.load(path)
    for k in ("imputer", "feature_cols", "base_model"):
        if k not in b:
            raise RuntimeError(f"Invalid bundle: missing '{k}'")
    return b


def _bundle_parts() -> Tuple[Any, List[str], Any, Optional[Any]]:
    b = _load_bundle()
    return b["imputer"], list(b["feature_cols"]), b["base_model"], b.get("calibrator", None)


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _get_as_of(conn: sqlite3.Connection, as_of: Optional[str]) -> datetime:
    if as_of:
        return _parse_time(as_of)
    row = conn.execute("SELECT MAX(event_at) FROM state_change WHERE event_at IS NOT NULL").fetchone()
    if not row or not row[0]:
        raise HTTPException(status_code=400, detail="state_change is empty (no event_at).")
    return _parse_time(row[0])


def _fetch_24h(conn: sqlite3.Connection, as_of: datetime) -> pd.DataFrame:
    t0 = (as_of - timedelta(hours=24)).isoformat()
    t1 = as_of.isoformat()
    sql = """
    SELECT event_at, statId, chgerId, prev_stat, new_stat
    FROM state_change
    WHERE event_at IS NOT NULL AND event_at >= ? AND event_at <= ?
    ORDER BY statId, chgerId, event_at
    """
    df = pd.read_sql_query(sql, conn, params=[t0, t1])
    if df.empty:
        return df

    df["event_at"] = pd.to_datetime(df["event_at"], errors="coerce")
    df["prev_stat"] = pd.to_numeric(df["prev_stat"], errors="coerce")
    df["new_stat"] = pd.to_numeric(df["new_stat"], errors="coerce")
    df = df.dropna(subset=["event_at", "prev_stat", "new_stat"]).copy()
    df["prev_stat"] = df["prev_stat"].astype(int)
    df["new_stat"] = df["new_stat"].astype(int)
    df["charger_key"] = df["statId"].astype(str) + "_" + df["chgerId"].astype(str)
    return df


def _build_last_event_features(df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """1 row per charger: last event <= as_of. Exclude current event from window counts."""
    if df.empty:
        return pd.DataFrame()

    out: List[Dict[str, Any]] = []

    for ck, g in df.groupby("charger_key", sort=False):
        g = g.sort_values("event_at")
        g2 = g[g["event_at"] <= as_of]
        if g2.empty:
            continue

        last = g2.iloc[-1]
        t = last["event_at"]

        if len(g2) >= 2:
            t_prev = g2.iloc[-2]["event_at"]
            gap_minutes = int((t - t_prev).total_seconds() / 60)
        else:
            gap_minutes = np.nan

        def counts(hours: int) -> Dict[str, int]:
            t0 = t - timedelta(hours=hours)
            gg = g2[(g2["event_at"] >= t0) & (g2["event_at"] < t)]  # exclude current
            return {
                f"n_events_{hours}h": int(len(gg)),
                f"to_9_{hours}h": int((gg["new_stat"] == 9).sum()),
                f"to_1_{hours}h": int((gg["new_stat"] == 1).sum()),
                f"to_4_{hours}h": int((gg["new_stat"] == 4).sum()),
                f"to_5_{hours}h": int((gg["new_stat"] == 5).sum()),
                f"to_bad_{hours}h": int(gg["new_stat"].isin(BAD_FOR_FEATURE).sum()),
                f"from9_to_normal_{hours}h": int(((gg["prev_stat"] == 9) & (gg["new_stat"].isin(NORMAL))).sum()),
                f"from1_to_normal_{hours}h": int(((gg["prev_stat"] == 1) & (gg["new_stat"].isin(NORMAL))).sum()),
                f"bad_to_normal_{hours}h": int(((gg["prev_stat"].isin(BAD_FOR_FEATURE)) & (gg["new_stat"].isin(NORMAL))).sum()),
            }

        row: Dict[str, Any] = {
            "charger_key": ck,
            "statId": str(last["statId"]),
            "chgerId": str(last["chgerId"]),
            "event_at": t.to_pydatetime().isoformat(),
            "prev_stat": int(last["prev_stat"]),
            "new_stat": int(last["new_stat"]),
            "gap_minutes": gap_minutes,
        }
        row.update(counts(6))
        row.update(counts(24))
        out.append(row)

    return pd.DataFrame(out)


def _predict(df_feat: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    imputer, feature_cols, base_model, calibrator = _bundle_parts()
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing feature columns: {missing[:20]} ... total={len(missing)}")

    X = df_feat[feature_cols].copy()
    X_imp = imputer.transform(X).astype(np.float32)

    p_raw = base_model.predict_proba(X_imp)[:, 1]
    p = calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1] if calibrator is not None else p_raw
    return p_raw, p


class DBScanRequest(BaseModel):
    db_path: str = Field(..., description="Path to sqlite DB (contains state_change)")
    as_of: Optional[str] = Field(default=None, description="ISO datetime; default=MAX(event_at)")
    threshold: float = Field(default=0.63, description="Return only p >= threshold")
    top_n: Optional[int] = Field(default=None, description="Optional cap after filtering; default=null (no cap)")
    score_col: Literal["p", "p_raw"] = Field(default="p", description="Use calibrated p or raw p for filtering/sorting")


@router.post("/anomaly_detection")
def db_scan(req: DBScanRequest):
    if not os.path.exists(req.db_path):
        raise HTTPException(status_code=400, detail=f"db_path not found: {req.db_path}")

    conn = sqlite3.connect(req.db_path)
    try:
        as_of = _get_as_of(conn, req.as_of)
        df = _fetch_24h(conn, as_of)
        df_feat = _build_last_event_features(df, as_of)

        if df_feat.empty:
            return {"as_of": as_of.isoformat(), "n_chargers": 0, "rows": []}

        p_raw, p = _predict(df_feat)
        out = df_feat[["charger_key", "statId", "chgerId", "event_at"]].copy()
        out["p_raw"] = p_raw
        out["p"] = p

        sc = req.score_col
        out = out[out[sc] >= float(req.threshold)].copy()
        out = out.sort_values(sc, ascending=False)

        if req.top_n is not None:
            out = out.head(int(req.top_n)).copy()

        return {
            "as_of": as_of.isoformat(),
            "threshold": req.threshold,
            "score_col": sc,
            "top_n": req.top_n,
            "n_chargers": int(out["charger_key"].nunique()) if not out.empty else 0,
            "rows": out.to_dict(orient="records"),
        }
    finally:
        conn.close()
