# Veltro — Prototype de scoring

Prototype de modèle de scoring pour identifier les cibles d'acquisition attractives.

## Structure du projet

```
prototype scoring veltro/
├── data/
│   ├── raw/                    # données brutes, jamais modifiées à la main
│   └── processed/              # données nettoyées/transformées si sauvegardées
├── notebooks/
│   └── Eda.ipynb               # exploration, jamais exécuté en production
├── src/
│   ├── generate_syntetic_data.py
│   ├── pipeline.py             # construction des pipelines (arbres / linéaire)
│   └── train.py                # entraînement + comparaison des modèles
├── models/                     # modèles entraînés sauvegardés (non versionnés Git)
├── tests/                      # tests unitaires du pipeline
├── reports/
│   └── canevas_eda.pdf
├── config.yaml                 # toute la configuration (colonnes, hyperparamètres, chemins)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```powershell
python -m src.train
```

## Décisions prises pendant l'EDA (résumé — détail dans `notebooks/Eda.ipynb`)

| Sujet | Décision | Justification |
|---|---|---|
| `annual_revenue`, `years_in_operation` | Yeo-Johnson + scaling, pipeline linéaire seulement | skew > 1 (2.23 et 1.66) ; n'améliore pas la corrélation avec la target, sert à stabiliser le modèle |
| Autres numériques | Scaling seul (linéaire), rien pour les arbres | skew modéré |
| `industry` | One-hot, les deux pipelines | chi² p=0.0000, V de Cramér=0.373 (lien substantiel) |
| `ownership_type` | One-hot, les deux pipelines | Lien faible-modéré mais présent |
| `geography` | One-hot arbres uniquement, exclue du linéaire | chi² p=0.48, non significatif |
| Target (`is_attractive_target`) | `class_weight='balanced'` sur les deux modèles | Déséquilibre ~70/30 |
| Métrique de comparaison | `average_precision` (PR-AUC) en priorité, `f1` et `roc_auc` en complément | Robuste au déséquilibre de classes, indépendant du seuil |

## Prochaines étapes

- [ ] Tuning des hyperparamètres du modèle retenu (`GridSearchCV` / `RandomizedSearchCV`)
- [ ] Calibration du score si utilisé comme probabilité (`CalibratedClassifierCV`)
- [ ] Choix du seuil de décision selon le coût métier (faux positif vs faux négatif)
- [ ] Évaluation finale sur `X_test` — une seule fois, jamais utilisé avant
- [ ] Explicabilité (SHAP) si le modèle retenu est une boîte noire
