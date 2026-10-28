# Fichier : labs/m0-setup/smoke_mlflow.py
"""Smoke test MLflow : enregistre un run avec paramètres, métriques et artefact."""

import os

import mlflow

# Serveur MLflow du cluster, accessible via make pf.
mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
)

mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="hello-mlops") as run:
    mlflow.log_param("framework", "scikit-learn")
    mlflow.log_metric("accuracy", 0.42)
    mlflow.log_metric("loss", 1.337)

    # Enregistrement de l'artefact dans le stockage configuré pour MLflow.
    mlflow.log_text("hello mlops\n", "hello.txt")

    print("Run logged successfully.")
    print(f"run_id       : {run.info.run_id}")
    print(f"artifact_uri : {run.info.artifact_uri}")