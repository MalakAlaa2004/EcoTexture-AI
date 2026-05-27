from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ecotexture_ai.config import MODELS_DIR
from ecotexture_ai.predict import load_assets, predict_image


@dataclass
class RuntimeStats:
    started_at: float
    requests_total: int = 0
    failures_total: int = 0
    last_latency_ms: float = 0.0


class PredictResponse(BaseModel):
    label: str
    label_ar: str | None = None
    confidence: float
    recommendation: str
    impact_note: str
    top_k: list[dict]
    bin_colour: str | None = None
    sdg_tags: list[str] | None = None
    latency_ms: float


app = FastAPI(title="EcoTexture AI API", version="1.0.0", docs_url="/docs")
stats = RuntimeStats(started_at=time.time())
model_bundle = None


@app.on_event("startup")
def startup() -> None:
    global model_bundle
    model_bundle = load_assets()


@app.get("/")
def root() -> dict:
    return {"service": "EcoTexture AI", "status": "ok"}


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "healthy",
        "uptime_sec": round(time.time() - stats.started_at, 2),
        "requests_total": stats.requests_total,
        "failures_total": stats.failures_total,
    }


@app.get("/readyz")
def readyz() -> dict:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "models_dir": str(MODELS_DIR)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    global model_bundle
    t0 = time.perf_counter()
    stats.requests_total += 1

    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = await file.read()
        np_bytes = np.frombuffer(raw, np.uint8)
        image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Invalid image bytes")

        model, centers, id_to_class = model_bundle
        result = predict_image(image_bgr, model, centers, id_to_class)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        stats.last_latency_ms = latency_ms

        return PredictResponse(
            label=result["label"],
            label_ar=result.get("label_ar"),
            confidence=float(result["confidence"]),
            recommendation=result["recommendation"],
            impact_note=result["impact_note"],
            top_k=result["top_k"],
            bin_colour=result.get("bin_colour"),
            sdg_tags=result.get("sdg_tags"),
            latency_ms=float(latency_ms),
        )
    except HTTPException:
        stats.failures_total += 1
        raise
    except Exception as exc:
        stats.failures_total += 1
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.get("/metrics")
def metrics() -> JSONResponse:
    payload = {
        "requests_total": stats.requests_total,
        "failures_total": stats.failures_total,
        "last_latency_ms": round(stats.last_latency_ms, 2),
        "uptime_sec": round(time.time() - stats.started_at, 2),
    }
    return JSONResponse(payload)
