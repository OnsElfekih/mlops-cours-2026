"""Détection du data drift : KS, chi-deux, PSI et Evidently."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
from evidently import Report
from evidently.presets import DataDriftPreset

from src.dataset_config import load_config


def calculate_psi(reference, current, bins=10):
    """Calcule le Population Stability Index."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if len(ref) == 0 or len(cur) == 0:
        return None

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))

    if len(edges) < 2:
        return 0.0 if np.array_equal(ref, cur) else None

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_hist = np.histogram(cur, bins=edges)[0] / len(cur)

    epsilon = 1e-6
    ref_hist = np.clip(ref_hist, epsilon, None)
    cur_hist = np.clip(cur_hist, epsilon, None)

    return float(
        np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
    )


def detect(reference, current, cfg, params):
    """Compare les distributions des variables explicatives."""
    results = []
    alpha = params["ks_alpha"]
    psi_threshold = params["psi_threshold"]

    for col in cfg.numeric:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur = pd.to_numeric(current[col], errors="coerce").dropna()

        ks = ks_2samp(ref, cur)
        psi = calculate_psi(ref, cur)

        std = float(ref.std())
        wasserstein = (
            float(wasserstein_distance(ref, cur) / std)
            if std > 0 else None
        )

        drift = bool(
            ks.pvalue < alpha
            or (psi is not None and psi > psi_threshold)
        )

        results.append({
            "feature": col,
            "type": "numeric",
            "ks_pvalue": float(ks.pvalue),
            "psi": psi,
            "wasserstein_normalized": wasserstein,
            "drift_detected": drift,
        })

    for col in cfg.categorical:
        ref = reference[col].fillna("__MISSING__").astype(str)
        cur = current[col].fillna("__MISSING__").astype(str)

        categories = sorted(set(ref) | set(cur))

        counts = np.array([
            [int((ref == value).sum()) for value in categories],
            [int((cur == value).sum()) for value in categories],
        ])

        _, pvalue, _, _ = chi2_contingency(counts)

        results.append({
            "feature": col,
            "type": "categorical",
            "chi2_pvalue": float(pvalue),
            "drift_detected": bool(pvalue < alpha),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--current", default="data/processed/test.csv")
    parser.add_argument("--output-dir", default="reports/drift")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    params = yaml.safe_load(
        Path("params.yaml").read_text(encoding="utf-8")
    )["drift"]

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)

    features = [*cfg.numeric, *cfg.categorical]

    results = detect(reference, current, cfg, params)

    drift_count = sum(item["drift_detected"] for item in results)
    drift_share = drift_count / len(results)
    drift_detected = drift_share >= params["drift_share"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "total_features": len(results),
        "drifted_features": drift_count,
        "drift_share": drift_share,
        "drift_detected": drift_detected,
        "features": results,
    }

    (output_dir / "drift.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        current_data=current[features],
        reference_data=reference[features],
    )

    snapshot.save_html(str(output_dir / "drift.html"))
    snapshot.save_json(str(output_dir / "evidently.json"))

    lines = [
        "# Rapport de Data Drift",
        "",
        f"- Observations de référence : {len(reference)}",
        f"- Observations actuelles : {len(current)}",
        f"- Variables analysées : {len(results)}",
        f"- Variables avec drift : {drift_count}",
        f"- Proportion de drift : {drift_share:.2%}",
        f"- Drift détecté : {drift_detected}",
        "",
        "## Résultats par variable",
        "",
    ]

    for item in results:
        status = "DRIFT" if item["drift_detected"] else "OK"
        lines.append(f"- {item['feature']} : {status}")

    (output_dir / "drift.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Variables analysées : {len(results)}")
    print(f"Variables avec drift : {drift_count}")
    print(f"Proportion de drift : {drift_share:.2%}")
    print(f"Drift détecté : {drift_detected}")
    print(f"Rapports enregistrés dans : {output_dir}")

    if drift_detected and args.fail_on_drift:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
