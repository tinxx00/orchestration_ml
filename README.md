# Projet fil rouge — Pipeline MLOps de classification : Prédiction de maladies cardiaques

> **Problématique** : Prédire si un patient présente une maladie cardiaque (`1`) ou non (`0`)
> à partir de données cliniques — permettre un dépistage rapide et non invasif.

---

## Dataset : Heart Disease UCI (Cleveland)

| Propriété         | Valeur                                                             |
|-------------------|--------------------------------------------------------------------|
| **Source**        | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) / [Kaggle](https://www.kaggle.com/code/mragpavank/heart-disease-uci/input) |
| **Auteurs**       | Janosi, Steinbrunn, Pfisterer, Detrano (1989)                      |
| **Licence**       | CC BY 4.0                                                          |
| **Instances**     | 303 patients                                                       |
| **Features**      | 13 variables cliniques                                             |
| **Cible**         | `target` — binaire (0 = pas de maladie, 1 = maladie présente)     |

### Description des colonnes

| Colonne      | Type          | Description                                                                 |
|--------------|---------------|-----------------------------------------------------------------------------|
| `age`        | Numérique     | Âge du patient (années)                                                     |
| `sex`        | Catégoriel    | Sexe (1 = homme, 0 = femme)                                                 |
| `cp`         | Catégoriel    | Type de douleur thoracique (1 = angor typique, 2 = atypique, 3 = non-angineux, 4 = asymptomatique) |
| `trestbps`   | Numérique     | Pression artérielle au repos (mm Hg)                                        |
| `chol`       | Numérique     | Cholestérol sérique (mg/dl)                                                 |
| `fbs`        | Catégoriel    | Glycémie à jeun > 120 mg/dl (1 = oui, 0 = non)                             |
| `restecg`    | Catégoriel    | Résultats ECG au repos (0 = normal, 1 = anomalie ST-T, 2 = hypertrophie ventriculaire gauche) |
| `thalach`    | Numérique     | Fréquence cardiaque maximale atteinte                                       |
| `exang`      | Catégoriel    | Angine induite par l'effort (1 = oui, 0 = non)                             |
| `oldpeak`    | Numérique     | Dépression ST induite par l'effort (relative au repos)                      |
| `slope`      | Catégoriel    | Pente du segment ST au pic d'effort (1 = montante, 2 = plate, 3 = descendante) |
| `ca`         | Numérique     | Nombre de vaisseaux majeurs colorés par fluoroscopie (0–3)                  |
| `thal`       | Catégoriel    | Thalassémie (3 = normal, 6 = défaut fixe, 7 = défaut réversible)           |
| **`target`** | **Cible**     | **0 = aucune maladie cardiaque, 1 = maladie cardiaque présente**            |

---

## Démarche pédagogique

1. **Problématique de classification binaire** : prédire la présence (`1`) ou l'absence (`0`)
   d'une maladie cardiaque chez un patient. Détecter les cas positifs permet d'orienter
   rapidement les patients vers un traitement adapté.

2. **Données** : le fichier `heart.csv` est placé dans `data/`. Il contient 303 lignes et
   14 colonnes (13 features + 1 cible).

3. **Suivi des étapes** : les `TODO (Sx-n)` dans le dossier `todo/` correspondent aux TP.
   Compléter uniquement ces marqueurs — le reste du code est fourni.

4. **Extension possible** : sections "Pour aller plus loin" des TP ou idées personnelles,
   après avoir réalisé tous les exercices obligatoires.

> Le seul fichier à adapter pour brancher ce dataset est `mlproject/config.py`
> (voir `tp/TP_S0_projet_personnel.md`). `data.py` et `features.py` lisent automatiquement
> la configuration.

---

## Mise en route

```bash
# 1. Installer les dépendances (uv requis)
make install

# 2. Rendre le package importable
export PYTHONPATH=.

# 3. Vérifier l'environnement
make doctor

# 4. Test rapide du pipeline de base (après TP S0 + S5)
python -m mlproject.train   # doit afficher f1=... roc_auc=...
```

---

## Structure du projet

```
orchestration_ml/
├── README.md                     ← ce fichier
├── Makefile                      ← cibles d'installation + TODO à compléter
├── pyproject.toml                ← dépendances et configuration (uv / hatchling)
├── data/
│   └── heart.csv                 ← dataset Heart Disease UCI  ← à placer ici
├── src/
│   └── churn/                    ← package Python principal
│       ├── config.py             ← configuration dataset          ← TP S0
│       ├── data.py               ← chargement + split             (fourni)
│       ├── features.py           ← pré-processing ColumnTransformer (fourni)
│       ├── evaluation.py         ← SHAP summary plot               (fourni)
│       ├── train.py              ← baseline MLflow                 ← TP S5
│       ├── train_optuna.py       ← optimisation Optuna + Registry  ← TP S6
│       ├── train_models.py       ← GridSearchCV + SHAP             ← TP S7
│       └── api.py                ← API FastAPI                     ← TP S12
├── frontend/
│   └── app.py                    ← frontend Streamlit              ← TP S14bis
├── docker/
│   ├── Dockerfile.train          ← image d'entraînement            ← TP S8
│   ├── Dockerfile.api            ← image API                       (fourni)
│   └── Dockerfile.frontend       ← image frontend                  (fourni)
├── docker-compose.yml            ← orchestration de la stack       ← TP S14
├── dags/
│   └── retrain_dag.py            ← DAG Airflow de ré-entraînement  ← TP S17
└── tests/                        ← tests pytest
```

---

## Feuille de route

| Séance  | Énoncé                            | Fichier à compléter              | Objectif                                     |
|---------|-----------------------------------|----------------------------------|----------------------------------------------|
| **S0**  | `tp/TP_S0_projet_personnel.md`    | `mlproject/config.py`            | Brancher `heart.csv`                         |
| **S5**  | `tp/TP_S5_mlflow.md`              | `mlproject/train.py`             | Suivi d'expériences MLflow                   |
| **S6**  | `tp/TP_S6_optuna.md`              | `mlproject/train_optuna.py`      | Optimisation Optuna + Model Registry         |
| **S7**  | `tp/TP_S7_automl_shap.md`         | `mlproject/train_models.py`      | Comparaison RF / XGBoost / LightGBM + SHAP   |
| **S8**  | `tp/TP_S8_docker.md`              | `docker/Dockerfile.train`        | Conteneuriser l'entraînement                 |
| **S12** | `tp/TP_S12_fastapi.md`            | `mlproject/api.py`               | Exposer le modèle via une API FastAPI        |
| **S14** | `tp/TP_S14_docker_compose.md`     | `docker-compose.yml`             | Orchestrer la stack (MLflow, API, frontend)  |
| **S14bis** | `tp/TP_S14_bis_streamlit.md`   | `frontend/app.py`                | Frontend de test Streamlit                   |
| **S17** | `tp/TP_S17_airflow.md`            | `dags/retrain_dag.py`            | Planifier le ré-entraînement avec Airflow    |

Toutes les commandes s'exécutent depuis la racine du projet (`PYTHONPATH=.`).

---

## Commandes utiles

```bash
make help            # liste toutes les cibles disponibles
make install         # crée le venv et installe les dépendances
make doctor          # vérifie l'environnement
make data            # prépare le dataset
make train           # entraîne la baseline  (C=1.0 MAX_ITER=1000)
make train-optuna    # optimisation Optuna   (N_TRIALS=30 CV=5)
make train-models    # GridSearchCV + SHAP   (CV=5 SCORING=roc_auc)
make mlflow          # démarre le serveur MLflow
make api             # lance l'API FastAPI    (host:port configurables)
make frontend        # lance le frontend Streamlit
make docker-build    # construit l'image d'entraînement
make docker-up       # démarre la stack complète
make check           # lint + types + tests
```

---

## Suivi GitHub (obligatoire)

- Committez et poussez votre travail **à chaque séance**, avec des messages clairs.
- Tenez ce `README.md` à jour avec vos résultats et observations.
- **Ajoutez l'enseignant comme collaborateur** :
  [github.com/lewishkpv](https://github.com/lewishkpv) — compte `lewishkpv`.
  *(Settings → Collaborators → Add people → `lewishkpv`)*

> ⚠️ Ne versionnez jamais de secrets (`.env`) ni d'artefacts lourds (`*.joblib`, `mlruns/`).
> Le `.gitignore` du projet les exclut déjà.

---

## Référence

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
*Heart Disease* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C52P4X>
