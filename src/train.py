"""
src/train.py

Charge les donnees, construit les pipelines (voir pipeline.py), les compare
par cross-validation, puis tune les hyperparametres du modele gagnant
(regression logistique, cf. resultats CV) avant evaluation finale sur X_test.

Usage (depuis la racine du projet) :
    python -m src.train
"""

import yaml
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from src.threshold_analysis import analyse_thresholds, find_threshold_for_target_precision
from src.pipeline import build_pipelines
import joblib


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

    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["split"]["random_state"],
    )
    metrics = config["model"]["scoring_metrics"]

    # -----------------------------------------------------------------
    # Etape 1 : comparaison des deux modeles a hyperparametres par defaut
    # -----------------------------------------------------------------
    for nom, pipeline in [("random_forest", pipeline_rf), ("logistic_regression", pipeline_logreg)]:
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=metrics)
        print(f"\n--- {nom} ---")
        for m in metrics:
            mean = scores[f"test_{m}"].mean()
            std = scores[f"test_{m}"].std()
            print(f"{m:20s} : {mean:.3f} (+/- {std:.3f})")

    # -----------------------------------------------------------------
    # Etape 2 : tuning des hyperparametres du modele gagnant
    # (regression logistique, cf. resultats CV : meilleur sur les 3 metriques
    # et plus stable / std plus faible)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Recherche des meilleurs hyperparametres (logistic_regression)")
    print("=" * 60)

    # Le nom du step dans le pipeline est "model" (cf. pipeline.py),
    # donc les parametres sont prefixes par "model__" pour GridSearchCV.
    param_grid = [
        {
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        {
            "model__penalty": ["l1"],
            "model__solver": ["liblinear"],
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
        },
        {
            "model__penalty": ["elasticnet"],
            "model__solver": ["saga"],
            "model__C": [0.01, 0.1, 1, 10],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
    ]

    # scoring principal = average_precision (le plus pertinent vu le
    # desequilibre ~70/30, cf. discussion EDA), on garde f1 et roc_auc
    # pour observation mais refit se fait sur average_precision.
    grid_search = GridSearchCV(
        estimator=pipeline_logreg,
        param_grid=param_grid,
        scoring=metrics,
        refit="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"\nMeilleurs hyperparametres : {grid_search.best_params_}")
    print(f"Meilleur score CV (average_precision) : {grid_search.best_score_:.3f}")

    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "models/best_model.pkl")

    # -----------------------------------------------------------------
    # Etape 3 : evaluation finale sur X_test (UNE SEULE FOIS, ici)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluation finale sur le jeu de test (jamais vu avant)")
    print("=" * 60)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"\naverage_precision (test) : {average_precision_score(y_test, y_proba):.3f}")
    print(f"roc_auc (test)           : {roc_auc_score(y_test, y_proba):.3f}")
    print("\nRapport de classification (test) :")
    print(classification_report(y_test, y_pred))


    analyse_thresholds(best_model, X_test, y_test)
    threshold = find_threshold_for_target_precision(best_model, X_test, y_test, target_precision=0.70)
    joblib.dump(threshold, "models/best_threshold.pkl")

   





if __name__ == "__main__":
    main()