from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.dataset_config import load_config

# Dossier de stockage des données préparées
PROCESSED = Path("data/processed")


def main() -> None:

    # Charger la configuration du dataset et les paramètres
    cfg = load_config()
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["prepare"]
    id_col = params["id_column"]

    # Charger le dataset brut et vérifier ses colonnes
    df = pd.read_csv(cfg.raw_path)
    cfg.validate_columns(df)

    # Ajouter un identifiant unique si nécessaire
    if id_col not in df.columns:
        df.insert(0, id_col, range(1, len(df) + 1))

    # Conserver les colonnes définies dans dataset.yaml
    df = df[[id_col, *cfg.columns]]

    # Encoder la cible pour réaliser un split stratifié
    y = cfg.encode_target(df[cfg.target])

    # Diviser les données : 80 % train et 20 % test
    train, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    # Créer le dossier et enregistrer les données
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)

    # Préparer les features pour Feast, sans la variable cible
    features = df[[id_col, *cfg.features]].copy()

    # Convertir les variables numériques en float64
    features[cfg.numeric] = features[cfg.numeric].astype("float64")

    # Ajouter une date fixe pour garantir la reproductibilité
    features["event_timestamp"] = pd.Timestamp(params["event_timestamp"])

    # Enregistrer les features au format Parquet
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    # Calculer la proportion de la classe positive dans train et test
    rate = (
        cfg.encode_target(train[cfg.target]).mean(),
        cfg.encode_target(test[cfg.target]).mean()
    )

    # Afficher le résultat de la préparation
    print(
        f"Train : {len(train)} / Test : {len(test)} · "
        f"classe '{cfg.positive_class}' : {rate[0]:.1%} / {rate[1]:.1%}"
    )


# Exécuter la fonction principale
if __name__ == "__main__":
    main()
