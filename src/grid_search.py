"""Run a grid search over model configurations and log results to MLflow."""

import os
import subprocess
import sys


def main():
    experiments = [
        # Dummy baseline: 1 configuration
        ["--model", "dummy"],

        # Logistic Regression: 4 C values × 2 class weights = 8
        *[
            ["--model", "logistic_regression", "--C", str(c),
             "--class_weight", weight]
            for c in [0.01, 0.1, 1, 10]
            for weight in ["balanced", "none"]
        ],

        # Random Forest: 3 estimator counts × 3 depths = 9
        *[
            ["--model", "random_forest", "--n_estimators", str(n),
             "--max_depth", depth]
            for n in [50, 100, 200]
            for depth in ["5", "none", "15"]
        ],

        # Gradient Boosting: 2 estimator counts × 2 learning rates = 4
        *[
            ["--model", "gradient_boosting", "--n_estimators", str(n),
             "--learning_rate", str(rate), "--max_depth", "3"]
            for n in [100, 200]
            for rate in [0.05, 0.1]
        ],
    ]

    env = os.environ.copy()
    env["LAUNCHED_BY"] = "grid_search"

    print(f"Lancement de {len(experiments)} configurations...", flush=True)

    for i, args in enumerate(experiments, start=1):
        print(f"\n[{i}/{len(experiments)}] {' '.join(args)}", flush=True)

        result = subprocess.run(
            [sys.executable, "-m", "src.train", *args],
            env=env,
            check=False,
        )

        if result.returncode != 0:
            print(f"⚠️ Échec de la configuration {i}", flush=True)

    print("\nGrid search terminé.", flush=True)


if __name__ == "__main__":
    main()
