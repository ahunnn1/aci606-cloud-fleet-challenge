from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db import DATABASE_URL, TelemetryRow, engine, get_session, init_db
from app.metrics import METRICS

logger = logging.getLogger("fleet")
ROOT = Path(__file__).resolve().parents[1]

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


# --------------------------------------------------------------------------- modelo
class ModelStore:
    """Modelo cargado una sola vez al iniciar el servicio."""

    pipeline = None
    features: list[str] = []
    classes: list[str] = []
    version: str = "unloaded"
    name: str | None = None
    error: str | None = None


MODEL = ModelStore()


def load_model() -> None:
    raw = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
    path = raw if raw.is_absolute() else ROOT / raw
    try:
        pipe = joblib.load(path)
        meta_path = path.with_name("model_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        trained_with = meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning("scikit-learn distinto: modelo entrenado con %s, instalado %s",
                           trained_with, sklearn.__version__)

        features = list(meta.get("features") or getattr(pipe, "feature_names_in_", []))
        if not features:
            raise ValueError("no se pudo determinar la lista de variables del modelo")

        MODEL.pipeline = pipe
        MODEL.features = features
        MODEL.classes = [str(c) for c in pipe.classes_]
        MODEL.version = os.environ.get("MODEL_VERSION") or meta.get("model_version", "v1")
        MODEL.name = meta.get("model_name")
        MODEL.error = None
        logger.info("Modelo cargado: %s (%s) desde %s", MODEL.name, MODEL.version, path)
    except Exception as exc:  # el servicio debe poder arrancar aunque falte el modelo
        MODEL.pipeline = None
        MODEL.error = f"{type(exc).__name__}: {exc}"
        logger.error("No se pudo cargar el modelo desde %s: %s", path, MODEL.error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()      # crea las tablas si no existen
    load_model()   # carga models/model.joblib (o MODEL_PATH)
    yield


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.4.0-metrics",
    description="Cloud Fleet Challenge API: persistent telemetry + ML inference.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """422 sin devolver el valor recibido (si fuera NaN, json.dumps fallaría -> 500)."""
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Mide todas las rutas: conteos, códigos de estado y latencia (incluye las 5xx)."""
    t0 = time.perf_counter()
    status = 500  # si call_next lanza una excepción no controlada, se cuenta como 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", None) or "unmatched"
        METRICS.record_request(request.method, route, status, (time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------- esquemas
class Telemetry(BaseModel):
    # NaN e Infinity se rechazan con 422. Los límites son físicamente generosos:
    # rechazan basura evidente sin descartar flotas con condiciones distintas.
    model_config = ConfigDict(allow_inf_nan=False)

    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float = Field(ge=0)
    x: float = Field(ge=-1e6, le=1e6)
    y: float = Field(ge=-1e6, le=1e6)
    speed: float = Field(ge=0, le=50)
    battery: float = Field(ge=0, le=100)
    motor_temp: float = Field(ge=-100, le=500)
    vibration: float = Field(ge=0, le=100)
    motor_current: float = Field(ge=0, le=1000)
    obstacle_distance: float = Field(ge=0, le=1000)


class PredictionRequest(Telemetry):
    pass


class PredictionResponse(BaseModel):
    robot_id: str
    predicted_status: Status
    confidence: float
    model_version: str
    inference_ms: float


def run_inference(sample: Telemetry) -> tuple[str, float, float]:
    """Devuelve (estado, confianza, milisegundos de inferencia)."""
    if MODEL.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    values = sample.model_dump()
    t0 = time.perf_counter()
    X = pd.DataFrame([[values[f] for f in MODEL.features]], columns=MODEL.features)
    proba = MODEL.pipeline.predict_proba(X)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    idx = int(proba.argmax())
    return MODEL.classes[idx], float(proba[idx]), elapsed_ms


# --------------------------------------------------------------------------- rutas
@app.get("/health")
def health() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    model_ok = MODEL.pipeline is not None
    return {
        "status": "ok" if (db_status == "ok" and model_ok) else "degraded",
        "database": db_status,
        "model": {"loaded": model_ok, "version": MODEL.version, "error": MODEL.error},
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "uptime_seconds": METRICS.snapshot()["uptime_seconds"],
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry, db: Session = Depends(get_session)) -> dict:
    """Guarda la muestra junto con la condición predicha (para el dashboard).

    Si el modelo falla, la telemetría se guarda igual con predicción vacía:
    perder datos de la flota sería peor que no tener predicción.
    """
    status = confidence = None
    try:
        status, confidence, _ = run_inference(sample)
    except Exception:
        logger.exception("Falló la inferencia durante la ingesta")
    db.add(TelemetryRow(**sample.model_dump(), predicted_status=status, confidence=confidence))
    db.commit()
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "predicted_status": status,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/robots")
def list_robots(db: Session = Depends(get_session)) -> list[dict]:
    """Última muestra de cada robot (incluye la condición predicha)."""
    last_ids = select(func.max(TelemetryRow.id)).group_by(TelemetryRow.robot_id)
    rows = db.scalars(
        select(TelemetryRow).where(TelemetryRow.id.in_(last_ids)).order_by(TelemetryRow.robot_id)
    ).all()
    return [r.to_dict() for r in rows]


@app.get("/robots/{robot_id}/history")
def robot_history(
    robot_id: str,
    limit: int | None = Query(default=None, ge=1, le=100_000),
    db: Session = Depends(get_session),
) -> list[dict]:
    """Historial completo (orden cronológico). `limit` opcional devuelve las últimas N."""
    exists = db.scalar(select(TelemetryRow.id).where(TelemetryRow.robot_id == robot_id).limit(1))
    if exists is None:
        raise HTTPException(status_code=404, detail="Robot not found")

    stmt = select(TelemetryRow).where(TelemetryRow.robot_id == robot_id)
    if limit is None:
        rows = db.scalars(stmt.order_by(TelemetryRow.id)).all()
    else:
        rows = list(reversed(db.scalars(stmt.order_by(TelemetryRow.id.desc()).limit(limit)).all()))
    return [r.to_dict() for r in rows]


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    """Inferencia pura: no escribe en la base de datos (segura bajo carga)."""
    status, confidence, ms = run_inference(req)
    METRICS.record_inference(ms)
    return PredictionResponse(
        robot_id=req.robot_id,
        predicted_status=status,
        confidence=round(confidence, 4),
        model_version=MODEL.version,
        inference_ms=round(ms, 3),
    )


@app.get("/metrics")
def metrics(db: Session = Depends(get_session)) -> dict:
    """Métricas operacionales.

    Los contadores de solicitudes/latencia/errores son del proceso (se reinician con el
    servicio); los datos de almacenamiento vienen de la base de datos y persisten.
    """
    total = db.scalar(select(func.count(TelemetryRow.id))) or 0
    robots = db.scalar(select(func.count(func.distinct(TelemetryRow.robot_id)))) or 0
    by_status = dict(
        db.execute(
            select(TelemetryRow.predicted_status, func.count())
            .where(TelemetryRow.predicted_status.is_not(None))
            .group_by(TelemetryRow.predicted_status)
        ).all()
    )
    return {
        **METRICS.snapshot(),
        "model_loaded": MODEL.pipeline is not None,
        "model_version": MODEL.version,
        "model_name": MODEL.name,
        "persistence": "SQLITE" if DATABASE_URL.startswith("sqlite") else "POSTGRESQL",
        "telemetry_rows_total": total,
        "robots_total": robots,
        "predictions_by_status": by_status,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = ROOT / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
