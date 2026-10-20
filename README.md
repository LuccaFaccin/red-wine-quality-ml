# Prédiction de la qualité des vins rouges par Machine Learning

## Présentation

Ce projet étudie la prédiction de la qualité des vins rouges à partir de leurs caractéristiques physico-chimiques, à l'aide de méthodes d'apprentissage supervisé **par régression**. Il suit les principales étapes d'un projet de Machine Learning : exploration des données, préparation, entraînement, comparaison, validation croisée, optimisation et interprétation des résultats.

L'analyse complète, le code et les visualisations sont disponibles dans le notebook [`notebooks/01_exploratory_analysis.ipynb`](notebooks/01_exploratory_analysis.ipynb).

## Jeu de données

Le projet utilise le jeu de données **Wine Quality – Red Wine**, issu de l'[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality).

- **1 599 observations** (vins rouges)
- **11 variables explicatives** physico-chimiques, dont `alcohol`, `volatile acidity`, `sulphates`, `pH` et `density`
- **Variable cible :** `quality`, une note de qualité comprise entre 3 et 8 dans cet échantillon
- **Valeurs manquantes :** aucune dans le fichier étudié

Les notes 5 et 6 représentent environ **82,6 %** des observations : la distribution de la variable cible est donc déséquilibrée.

## Méthodologie

1. **Exploration des données :** structure du jeu de données, statistiques descriptives, distribution des notes et corrélations.
2. **Préparation :** séparation des variables explicatives et de la cible, puis division en données d'entraînement (80 %) et de test (20 %), avec `random_state=42`.
3. **Entraînement :** comparaison de trois algorithmes de régression :
   - Régression linéaire (`LinearRegression`)
   - Forêt aléatoire (`RandomForestRegressor`)
   - Gradient Boosting (`GradientBoostingRegressor`)
4. **Évaluation :** MAE, RMSE et R², puis validation croisée à 5 folds sur les données d'entraînement.
5. **Optimisation :** recherche de 24 combinaisons d'hyperparamètres du Random Forest avec `GridSearchCV` (5 folds, soit 120 ajustements).
6. **Interprétation :** importance des variables selon le Random Forest et par permutation, puis analyse des prédictions et des résidus.

## Résultats

### Comparaison par validation croisée

| Modèle | MAE moyen | Écart-type |
|---|---:|---:|
| Régression linéaire | 0,5057 | 0,0178 |
| **Random Forest** | **0,4310** | 0,0207 |
| Gradient Boosting | 0,4798 | 0,0159 |

Le **Random Forest** obtient la plus faible erreur absolue moyenne en validation croisée parmi les modèles testés.

### Modèle retenu : Random Forest optimisé

Meilleurs hyperparamètres trouvés : `n_estimators=100`, `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=1`.

| Métrique sur l'ensemble de test | Résultat |
|---|---:|
| **MAE** | **0,4224** |
| RMSE | 0,5487 |
| R² | 0,5392 |
| Prédictions avec erreur absolue ≤ 0,5 | 63,44 % |
| Prédictions avec erreur absolue ≤ 1,0 | 93,12 % |

L'optimisation apporte une **amélioration très faible** par rapport au Random Forest initial.

Les variables les plus importantes selon les deux méthodes étudiées sont **`alcohol`**, **`sulphates`** et **`volatile acidity`**. Ces importances décrivent le comportement prédictif du modèle et ne démontrent pas de causalité.

## Limites et perspectives

- Les notes extrêmes sont peu représentées ; le modèle tend à surestimer les faibles notes et à sous-estimer les notes élevées.
- Le R² d'environ 0,54 indique qu'une partie importante de la variabilité reste inexpliquée.
- L'ensemble de test a été consulté lors des premières comparaisons exploratoires : l'évaluation finale n'est donc pas totalement indépendante des décisions prises au cours du projet.
- Des améliorations possibles comprennent une validation croisée imbriquée, l'étude d'autres algorithmes (XGBoost, CatBoost), un enrichissement des données et une approche de classification ordinale.

## Structure du projet

```text
red-wine-quality-ml/
├── data/
│   └── winequality-red.csv
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── requirements.txt
└── README.md
```

## Installation et exécution

Prérequis : **Python 3** et **Git**.

```bash
git clone https://github.com/LuccaFaccin/red-wine-quality-ml.git
cd red-wine-quality-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Dans JupyterLab, ouvrir `notebooks/01_exploratory_analysis.ipynb`, puis exécuter les cellules dans l'ordre. Le notebook lit le fichier CSV local avec le séparateur `;`.

> Remarque : les commandes d'activation de l'environnement virtuel ci-dessus concernent Linux/WSL/macOS. Vérifier que `requirements.txt` contient bien les dépendances nécessaires, notamment `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` et `jupyterlab`.

## Technologies utilisées

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · JupyterLab · Git/GitHub

## Source des données

Cortez, P., Cerdeira, A., Almeida, F., Matos, T. et Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553. [Jeu de données UCI](https://archive.ics.uci.edu/dataset/186/wine+quality).
