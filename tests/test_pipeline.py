 


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from src.pipeline import build_pipelines
from src.train import load_config
import pandas as pd




def test_build_pipelines_retourne_deux_pipelines():
    config = load_config()
    pipeline_rf, pipeline_logreg = build_pipelines(config)
    assert pipeline_rf is not None , "on espere le pipeline n'est pas vide "
    assert pipeline_logreg is not None , "on espere le pipeline n'est pas vide "

def test_build_pipelines_types():
    config = load_config()
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
    pipeline_rf, pipeline_logreg = build_pipelines(config)
    pipeline_logreg.fit(X, y)
    preprocessing = pipeline_logreg.named_steps["preprocessing"]
    X_transformed = preprocessing.transform(X)
    assert X_transformed.shape[1] == 19, f"Attendu 19 colonnes, obtenu {X_transformed.shape[1]}"
