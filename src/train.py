"""
src/train.py

Charge les donnees, construit les pipelines (voir pipeline.py), les compare
par cross-validation, et affiche les resultats.

Usage (depuis la racine du projet) :
    python -m src.train
"""

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

from src.pipeline import build_pipelines


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # --- Chargement des donnees ---
    df = pd.read_csv(config["data"]["raw_path"])

    id_col = config["data"].get("id_column")
    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])

    target = config["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    # --- Split stratifie ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        stratify=y,
        random_state=config["split"]["random_state"],
    )

    # --- Construction des pipelines ---
    pipeline_rf, pipeline_logreg = build_pipelines(config)

    # --- Comparaison par cross-validation (sur X_train uniquement) ---
    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["split"]["random_state"],
    )
    metrics = config["model"]["scoring_metrics"]

    for nom, pipeline in [("random_forest", pipeline_rf), ("logistic_regression", pipeline_logreg)]:
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=metrics)
        print(f"\n--- {nom} ---")
        for m in metrics:
            mean = scores[f"test_{m}"].mean()
            std = scores[f"test_{m}"].std()
            print(f"{m:20s} : {mean:.3f} (+/- {std:.3f})")

    # Rappel : X_test n'est PAS utilise ici. Il ne doit servir qu'une seule fois,
    # a la toute fin, une fois le modele et les hyperparametres definitivement choisis.


if __name__ == "__main__":
    main()
