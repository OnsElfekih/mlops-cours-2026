"""Évalue le modèle local et écrit les métriques dans reports/metrics.json."""

import json
from pathlib import Path

import joblib
import yaml
from sklearn.metrics import confusion_matrix

from src.dataset_config import load_config
from src.train import MODEL_DIR, TEST_CSV, compute_metrics

METRICS_JSON = Path("reports/metrics.json")


def main() -> None:
    cfg = load_config()
    X_test, y_test = cfg.read_xy(TEST_CSV)

    # Charger le modèle local.
    model = joblib.load(MODEL_DIR / "model.joblib")

    # Charger les métadonnées du modèle.
    with open(MODEL_DIR / "metadata.yaml", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    threshold = float(metadata["decision_threshold"])

    # Calculer les probabilités et les métriques.
    proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba, threshold, cfg)

    # Calculer la matrice de confusion.
    tn, fp, fn, tp = confusion_matrix(
        y_test, (proba >= threshold).astype(int)
    ).ravel()

    # Préparer le rapport.
    report = {
        **{key: round(float(value), 4) for key, value in metrics.items()},
        "decision_threshold": round(threshold, 4),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "n_test": int(len(y_test)),
    }

    # Enregistrer les métriques dans un fichier JSON.
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"{cfg.primary_metric} = "
        f"{report[cfg.primary_metric]} -> {METRICS_JSON}"
    )


if __name__ == "__main__":
    main()
