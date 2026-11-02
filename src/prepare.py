"""Prepare the German Credit dataset for training and Feast."""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.dataset_config import load_config


ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_config()

    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    raw_path = ROOT / config.raw_path
    df = pd.read_csv(raw_path)
    config.validate_columns(df)

    id_column = params["prepare"]["id_column"]

    if id_column not in df.columns:
        df.insert(0, id_column, range(1, len(df) + 1))

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df[config.target],
    )

    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    features_df = df[[id_column] + config.features].copy()
    features_df["event_timestamp"] = pd.Timestamp(
        params["prepare"]["event_timestamp"]
    )

    features_df.to_parquet(
        processed_dir / "features.parquet",
        index=False,
    )

    positive_class = str(config.positive_class)

    train_rate = (
        train_df[config.target].astype(str) == positive_class
    ).mean()

    test_rate = (
        test_df[config.target].astype(str) == positive_class
    ).mean()

    print(f"Train: {len(train_df)} lignes")
    print(f"Test: {len(test_df)} lignes")
    print(f"Features Feast: {len(features_df)} lignes")
    print(
        f"Taux de classe '{positive_class}' — "
        f"train: {train_rate:.1%}, test: {test_rate:.1%}"
    )


if __name__ == "__main__":
    main()
