"""
src/pipeline.py

Construction des deux pipelines de scoring (arbres / lineaire) a partir des
decisions documentees dans config.yaml (issues de l'EDA, voir notebooks/Eda.ipynb).

Usage :
    from src.pipeline import build_pipelines
    pipeline_rf, pipeline_logreg = build_pipelines(config)
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_pipelines(config: dict):
    """
    Construit les pipelines "arbres" et "lineaire" a partir de la config.

    Parametres
    ----------
    config : dict
        Contenu charge depuis config.yaml (voir clef "columns" et "model").

    Retourne
    --------
    (pipeline_rf, pipeline_logreg) : tuple de sklearn.pipeline.Pipeline
    """
    cols = config["columns"]
    model_cfg = config["model"]

    cols_num_asymetriques = cols["numeric_skewed"]
    cols_num_normales = cols["numeric_normal"]
    cols_num_toutes = cols_num_asymetriques + cols_num_normales

    cols_cat_signal = cols["categorical_signal"]
    cols_cat_sans_signal = cols["categorical_no_signal"]

    class_weight = model_cfg.get("class_weight", "balanced")
    random_state = config.get("split", {}).get("random_state", 42)

    # -----------------------------------------------------------------
    # Pipeline "arbres" (Random Forest) - preparation minimale
    # -----------------------------------------------------------------
    cols_cat_rf = cols_cat_signal + cols_cat_sans_signal

    preprocessing_rf = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), cols_num_toutes),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cols_cat_rf),
    ])

    pipeline_rf = Pipeline([
        ("preprocessing", preprocessing_rf),
        ("model", RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
        )),
    ])

    # -----------------------------------------------------------------
    # Pipeline "lineaire" (Regression logistique) - preparation complete
    # geography est volontairement exclue (cf. config.yaml, p=0.48)
    # -----------------------------------------------------------------
    cols_cat_logreg = cols_cat_signal

    preprocessing_logreg = ColumnTransformer([
        ("num_asym", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("power_transform", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
        ]), cols_num_asymetriques),

        ("num_normal", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), cols_num_normales),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cols_cat_logreg),
    ])

    pipeline_logreg = Pipeline([
        ("preprocessing", preprocessing_logreg),
        ("model", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
        )),
    ])

    return pipeline_rf, pipeline_logreg
