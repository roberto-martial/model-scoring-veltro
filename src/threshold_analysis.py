"""
src/threshold_analysis.py

Analyse precision / recall / f1 a travers differents seuils de decision,
pour remplacer le seuil par defaut (0.5) par un seuil adapte au contexte
metier (cout d'un faux positif vs cout d'un faux negatif pour Veltro).

Usage :
    from src.threshold_analysis import analyse_thresholds, best_model  (deja fit)
    analyse_thresholds(best_model, X_test, y_test)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve


def analyse_thresholds(model, X_test, y_test, thresholds=None):
    """
    Calcule precision, recall, f1 pour une serie de seuils de decision,
    et affiche un tableau recapitulatif pour aider au choix du seuil.

    Parametres
    ----------
    model : pipeline sklearn deja entraine (avec predict_proba)
    X_test, y_test : jeu de test (jamais utilise pour l'entrainement/tuning)
    thresholds : liste de seuils a tester (par defaut : 0.1 a 0.9 par pas de 0.05)

    Retourne
    --------
    pd.DataFrame avec une ligne par seuil teste
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)

    y_proba = model.predict_proba(X_test)[:, 1]

    resultats = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)

        # zero_division=0 pour eviter les warnings/erreurs si un seuil
        # extreme ne produit aucune prediction positive
        precision = precision_score(y_test, y_pred_t, zero_division=0)
        recall = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        n_flagged = y_pred_t.sum()

        resultats.append({
            "threshold": round(t, 2),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "n_deals_flagged": int(n_flagged),
        })

    df_resultats = pd.DataFrame(resultats)

    print("\n" + "=" * 70)
    print("Precision / Recall / F1 par seuil de decision")
    print("=" * 70)
    print(df_resultats.to_string(index=False))

    # Seuil qui maximise le f1 (compromis equilibre, a titre indicatif)
    meilleur_f1 = df_resultats.loc[df_resultats["f1"].idxmax()]
    print(f"\nSeuil maximisant le f1-score : {meilleur_f1['threshold']} "
          f"(precision={meilleur_f1['precision']}, recall={meilleur_f1['recall']})")

    return df_resultats


def find_threshold_for_target_precision(model, X_test, y_test, target_precision=0.70):
    """
    Trouve le plus petit seuil qui permet d'atteindre une precision cible.
    Utile si le cout d'un faux positif est eleve (ex : temps analyste
    engage sur un deal qui n'etait pas vraiment attractif).

    Parametres
    ----------
    target_precision : precision minimale souhaitee (ex : 0.70 = 70%)
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, seuils_pr = precision_recall_curve(y_test, y_proba)

    # precision_recall_curve retourne un point de plus que seuils_pr
    # (le dernier point correspond a seuil=1.0, sans seuil associe)
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    candidats = [(t, p, r) for t, p, r in zip(seuils_pr, precisions, recalls) if p >= target_precision]

    if not candidats:
        print(f"\nAucun seuil n'atteint une precision de {target_precision:.0%} sur ce jeu de test.")
        return None

    # Le premier candidat (plus petit seuil) maximise le recall
    # parmi ceux qui respectent la contrainte de precision
    seuil_choisi, precision_obtenue, recall_obtenu = candidats[0]

    print(f"\nSeuil minimal pour precision >= {target_precision:.0%} : {seuil_choisi:.3f}")
    print(f"  -> precision obtenue : {precision_obtenue:.3f}")
    print(f"  -> recall obtenu     : {recall_obtenu:.3f}")

    return seuil_choisi


if __name__ == "__main__":
    # Exemple d'usage autonome : necessite d'avoir deja le best_model
    # entraine (voir train.py). A adapter/importer selon ton usage reel.
    print("Ce module est concu pour etre importe apres l'entrainement du modele final.")
    print("Voir train.py pour l'integration (best_model, X_test, y_test).")