"""Train a credit-risk model and log results to MLflow."""

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.dataset_config import load_config


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
MODEL_DIR = ROOT / "models" / "model"


def compute_metrics(y_true, probabilities, threshold, config):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "business_cost": float(config.business_cost(y_true, predictions)),
    }


def build_model(model_name, args, config):
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=args.C,
            class_weight=args.class_weight,
            max_iter=1000,
            random_state=args.random_state,
        )

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight=args.class_weight,
            random_state=args.random_state,
            n_jobs=-1,
        )

    elif model_name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )

    elif model_name == "dummy":
        classifier = DummyClassifier(strategy="prior")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            ("preprocessor", config.build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "dummy",
        ],
    )
    parser.add_argument("--C", type=float)
    parser.add_argument("--class_weight")
    parser.add_argument("--n_estimators", type=int)
    parser.add_argument("--max_depth")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--random_state", type=int)

    args = parser.parse_args()

    if args.class_weight == "none":
        args.class_weight = None

    if args.max_depth == "none":
        args.max_depth = None
    elif args.max_depth is not None:
        args.max_depth = int(args.max_depth)

    config = load_config()

    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        defaults = yaml.safe_load(file)["train"]

    args.model = args.model or defaults["model"]
    args.C = args.C if args.C is not None else defaults["C"]
    args.class_weight = (
        args.class_weight
        if args.class_weight is not None
        else defaults["class_weight"]
    )
    args.n_estimators = (
        args.n_estimators
        if args.n_estimators is not None
        else defaults["n_estimators"]
    )
    args.max_depth = (
        args.max_depth
        if args.max_depth is not None
        else defaults["max_depth"]
    )
    args.learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else defaults["learning_rate"]
    )
    args.random_state = (
        args.random_state
        if args.random_state is not None
        else defaults["random_state"]
    )

    X_train, y_train = config.read_xy(TRAIN_CSV)
    X_test, y_test = config.read_xy(TEST_CSV)

    model = build_model(args.model, args, config)

    mlflow.set_experiment("scoring")

    with mlflow.start_run(run_name=args.model) as run:
        mlflow.set_tags(
            {
                "model_type": args.model,
                "dataset": config.name,
                "launched_by": os.environ.get("LAUNCHED_BY", "manual"),
            }
        )

        mlflow.log_params(
            {
                "model": args.model,
                "C": args.C,
                "class_weight": args.class_weight,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "random_state": args.random_state,
            }
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        # Adjust the threshold for balanced class weights.
        weight_ratio = 1.0

        if (
            args.model in ("logistic_regression", "random_forest")
            and args.class_weight == "balanced"
        ):
            counts = np.bincount(y_train)
            weight_ratio = counts[0] / counts[1]

        threshold = config.cost_threshold(weight_ratio)
        metrics = compute_metrics(
            y_test, probabilities, threshold, config
        )

        mlflow.log_metrics(metrics)
        mlflow.log_param("decision_threshold", threshold)

        signature = infer_signature(
            X_train, model.predict_proba(X_train)
        )
        input_example = X_train.head(5)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_DIR / "model.joblib")

        metadata = {
            "model_type": args.model,
            "decision_threshold": float(threshold),
            "run_id": run.info.run_id,
            "metrics": metrics,
        }

        with open(
            MODEL_DIR / "metadata.yaml", "w", encoding="utf-8"
        ) as file:
            yaml.safe_dump(metadata, file)

        print(f"Modèle : {args.model}")
        print(f"Run ID : {run.info.run_id}")
        print(f"Seuil : {threshold:.4f}")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
