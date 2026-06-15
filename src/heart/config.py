"""Configuration du projet Heart Disease UCI — seul fichier à adapter (TP S0)."""
from pathlib import Path

# Racine du projet (3 niveaux au-dessus de src/churn/config.py)
ROOT = Path(__file__).resolve().parent.parent.parent

# ===========================================================================
# TODO (S0-1) : chemin vers votre fichier CSV
# ===========================================================================
DATA_PATH = ROOT / "data" / "heart.csv"

# ===========================================================================
# TODO (S0-2) : nom de la colonne cible (binaire 0/1)
# ===========================================================================
TARGET = "target"
# 0 = aucune maladie cardiaque  |  1 = maladie cardiaque présente

# ===========================================================================
# TODO (S0-3) : colonnes numériques
# ===========================================================================
NUMERIC_FEATURES = [
    "age",        # âge du patient (années)
    "trestbps",   # pression artérielle au repos (mm Hg)
    "chol",       # cholestérol sérique (mg/dl)
    "thalach",    # fréquence cardiaque maximale atteinte
    "oldpeak",    # dépression ST induite par l'effort
    "ca",         # nombre de vaisseaux colorés par fluoroscopie (0-3)
]

# ===========================================================================
# TODO (S0-4) : colonnes catégorielles (liste vide si aucune)
# ===========================================================================
CATEGORICAL_FEATURES = [
    "sex",      # 1 = homme, 0 = femme
    "cp",       # type de douleur thoracique (1-4)
    "fbs",      # glycémie à jeun > 120 mg/dl (1 = oui, 0 = non)
    "restecg",  # résultats ECG au repos (0/1/2)
    "exang",    # angine induite par l'effort (1 = oui, 0 = non)
    "slope",    # pente du segment ST au pic d'effort (1/2/3)
    "thal",     # thalassémie (3 = normal, 6 = fixe, 7 = réversible)
]

# ===========================================================================
# MLflow
# ===========================================================================
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT   = "heart-disease-baseline"
MODEL_NAME          = "heart-disease-classifier"

# ===========================================================================
# Split train/test
# ===========================================================================
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ===========================================================================
# Hyper-paramètres baseline (LogisticRegression)
# ===========================================================================
C        = 1.0
MAX_ITER = 1000

# ===========================================================================
# Chemins d'artefacts
# ===========================================================================
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.joblib"
