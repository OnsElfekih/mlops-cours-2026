"""API de scoring : sert le modèle ONNX exporté à l'étape 1.

Le schéma d'entrée est construit à partir de dataset.yaml : aucune colonne n'est écrite
ici, changer de dataset ne demande aucune modification de ce fichier.

Lancement local (depuis la racine du repo) :
    uv run python -m uvicorn app.serve:app --app-dir packaging --port 8000
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.dataset_config import load_config  # noqa: E402

API_VERSION = os.getenv("APP_VERSION", "1.0.0")
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "packaging" / "model" / "model.onnx"))

cfg = load_config(os.getenv("DATASET_CONFIG", ROOT / "dataset.yaml"))
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
meta_path = MODEL_PATH.with_name("metadata.json")
metadata: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}

THRESHOLD = float(os.getenv("DECISION_THRESHOLD", metadata.get("threshold", cfg.cost_threshold())))
MODEL_VERSION = str(metadata.get("model_version", "unknown"))
MODEL_KEYS = ("model_name", "model_alias", "model_version", "run_id", "onnx_sha256", "exported_at")

Numeric = Annotated[float | None, Field(strict=True, allow_inf_nan=False)]
Categorical = Annotated[str | None, Field(strict=True, min_length=1)]
Instance = create_model(
    "Instance",
    __config__=ConfigDict(extra="forbid"),
    **{c: (Numeric, ...) for c in cfg.numeric},
    **{c: (Categorical, ...) for c in cfg.categorical},
)


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(..., min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    probability: float
    prediction: int


class PredictResponse(BaseModel):
    model_version: str
    positive_class: str
    threshold: float
    predictions: list[Prediction]


app = FastAPI(title=f"Scoring API — {cfg.name}", version=API_VERSION)


def score(rows: list[dict[str, Any]]) -> np.ndarray:
    """Probabilité de la classe positive pour chaque demande."""
    feed = {c: np.array([[r[c]] for r in rows], dtype=np.float64) for c in cfg.numeric}
    feed |= {c: np.array([[r[c] or ""] for r in rows], dtype=object) for c in cfg.categorical}
    (proba,) = session.run(["probabilities"], feed)
    return proba[:, 1]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/metadata")
def get_metadata():
    return {
        "api_version": API_VERSION,
        "dataset": cfg.name,
        "target": cfg.target,
        "positive_class": str(cfg.positive_class),
        "threshold": THRESHOLD,
        "features": {"numeric": cfg.numeric, "categorical": cfg.categorical},
        "model": {k: metadata.get(k) for k in MODEL_KEYS},
    }

@app.get("/version")
def version():
    # Lab 5 : quelle version a répondu ? (canary, rollback, tests de fumée du Lab 6)
    return {"version": API_VERSION, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        proba = score([i.model_dump() for i in req.instances])
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return PredictResponse(
        model_version=MODEL_VERSION,
        positive_class=str(cfg.positive_class),
        threshold=THRESHOLD,
        predictions=[
            Prediction(probability=float(p), prediction=int(p >= THRESHOLD)) for p in proba
        ],
    )
