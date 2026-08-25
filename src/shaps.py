
import joblib
import pandas as pd 
import shap
from sklearn import preprocessing
from src.train import load_config
from sklearn.model_selection import train_test_split
from src.score_new_deal import dict_deal_2



#data preparation 
best_model = joblib.load("models/best_model.pkl")
preprocessing = best_model.named_steps["preprocessing"]

def entrain(): 
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
    return X_train
X_train = entrain()


### now we will just do the shapley test to see which variable push the result to a certain direction

def shapley_analysis(best_model, X_train):

    X_transformed = preprocessing.transform(X_train)
    shaper = shap.LinearExplainer(best_model.named_steps["model"], X_transformed)

    return shaper

shaper = shapley_analysis(best_model, X_train)
joblib.dump(shaper, "models/best_shaper.pkl")

def shapley_test(dict_deal : dict, shaper): 
    dic ={}
    new_deal_df = pd.DataFrame([dict_deal])
    new_deal_transformed = preprocessing.transform(new_deal_df)
    shap_values = shaper(new_deal_transformed)
    print(shap_values)
    noms_features = preprocessing.get_feature_names_out()
    resultats = list(zip(noms_features, shap_values.values[0], shap_values.data[0]))
    resultats_tries = sorted(resultats, key=lambda x: abs(x[1]), reverse=True)

    print(f"\nScore de base : {shap_values.base_values[0]:.3f}\n")
    for nom, contribution, valeur in resultats_tries:
        signe = "+" if contribution >= 0 else ""
        print(f"{nom:30s} {signe}{contribution:.3f}")
        dic[nom] = float(contribution)

    return dic

if __name__ == "__main__":
    shapley_test(dict_deal_2, shaper)