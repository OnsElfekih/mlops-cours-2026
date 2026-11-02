"""Register the best MLflow run and manage champion/challenger aliases."""

import argparse
import hashlib
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.dataset_config import load_config

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "scoring-model"


def get_data_md5():
    """Get the DVC-tracked dataset checksum, when available."""
    lock_path = ROOT / "dvc.lock"
    if not lock_path.exists():
        return "unknown"

    with open(lock_path, encoding="utf-8") as file:
        lock = yaml.safe_load(file)

    for stage in lock.get("stages", {}).values():
        for output in stage.get("outs", []):
            if output.get("path") == "data/raw/dataset.csv":
                return output.get("md5", "unknown")

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="MLflow run ID to register")
    parser.add_argument("--filter", dest="filter_string",
                        help="MLflow filter for selecting a run")
    args = parser.parse_args()

    config = load_config()
    metric_name = config.primary_metric
    threshold = config.promotion_threshold

    client = MlflowClient()
    experiment = client.get_experiment_by_name("scoring")

    if experiment is None:
        raise RuntimeError("L'expérience MLflow 'scoring' est introuvable.")

    if args.run_id:
        run = client.get_run(args.run_id)
        candidates = [run]
    else:
        filter_string = args.filter_string or ""
        candidates = client.search_runs(
            [experiment.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=100,
        )

    candidates = [
        run for run in candidates
        if metric_name in run.data.metrics
    ]

    if not candidates:
        raise RuntimeError("Aucun run valide trouvé pour l'enregistrement.")

    best_run = max(
        candidates,
        key=lambda run: run.data.metrics[metric_name],
    )
    score = best_run.data.metrics[metric_name]
    run_id = best_run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    registered = client.search_registered_models(
        filter_string=f"name = '{MODEL_NAME}'"
    )

    if not registered:
        client.create_registered_model(MODEL_NAME)

    existing_versions = client.search_model_versions(
        f"name = '{MODEL_NAME}'"
    )
    existing_version = next(
        (v for v in existing_versions if v.run_id == run_id),
        None,
    )

    if existing_version:
        version = existing_version.version
        print(f"Run déjà enregistré : version {version}")
    else:
        result = mlflow.register_model(model_uri, MODEL_NAME)
        version = result.version
        print(f"Nouvelle version enregistrée : {version}")

    client.set_model_version_tag(
        MODEL_NAME, version, "primary_metric", metric_name
    )
    client.set_model_version_tag(
        MODEL_NAME, version, "validation_status",
        "passed" if score >= threshold else "below_threshold"
    )
    client.set_model_version_tag(
        MODEL_NAME, version, "data_md5", get_data_md5()
    )

    client.set_registered_model_alias(
        MODEL_NAME, "challenger", version
    )

    if score >= threshold:
        client.set_registered_model_alias(
            MODEL_NAME, "champion", version
        )
        print("Le modèle devient champion.")
    else:
        print(
            f"Score {score:.4f} inférieur au seuil "
            f"{threshold:.4f} : champion inchangé."
        )

    print(f"Modèle : {MODEL_NAME}")
    print(f"Version : {version}")
    print(f"Run ID : {run_id}")
    print(f"{metric_name} : {score:.4f}")
    print(f"Seuil de promotion : {threshold:.4f}")
    print(f"MD5 du dataset : {get_data_md5()}")


if __name__ == "__main__":
    main()
