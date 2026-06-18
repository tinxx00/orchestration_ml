"""Frontend Streamlit — Heart Disease Classifier (thème clair pastel, navigation latérale)."""
from __future__ import annotations

import json
import os
from datetime import datetime

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

API_URL        = os.getenv("API_URL",        "http://localhost:8000")
MLFLOW_URL     = os.getenv("MLFLOW_URL",     "http://localhost:5000")
AIRFLOW_URL    = os.getenv("AIRFLOW_URL",     "http://localhost:8080")
API_PUBLIC_URL     = os.getenv("API_PUBLIC_URL",     API_URL)
MLFLOW_PUBLIC_URL  = os.getenv("MLFLOW_PUBLIC_URL",  MLFLOW_URL)
AIRFLOW_PUBLIC_URL = os.getenv("AIRFLOW_PUBLIC_URL", AIRFLOW_URL)
AIRFLOW_USER     = os.getenv("AIRFLOW_USER",     "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
RETRAIN_DAG_ID   = os.getenv("RETRAIN_DAG_ID",   "model_retraining")
EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT", "heart-disease-baseline")

AUTHOR = "Tinhinane ISSAD"
GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/tinxx00/orchestration_ml")

# ─── Palette pastel bleue ────────────────────────────────────────────────────────
INK      = "#2f4a66"
MUTED    = "#8595a8"
BLUE     = "#5b9bd5"
MINT     = "#5cb8a8"
CORAL    = "#e88a8a"
AMBER    = "#e9b872"
SKY      = "#6fa8dc"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #eef5fc; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #dcecfb 0%, #eaf4fc 100%);
    border-right: 1px solid #d8e6f3;
}
[data-testid="stSidebar"] * { color: #34536e; }

h1, h2, h3, h4 { color: #2f4a66; }

/* Carte auteur sidebar */
.author-card {
    background: linear-gradient(135deg, #5b9bd5 0%, #7ec0ef 100%);
    border-radius: 16px; padding: 1.1rem 1.2rem; margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(91,155,213,0.25);
}
.author-card * { color: #ffffff !important; }
.author-card .title { font-size: 1.05rem; font-weight: 800; }
.author-card .name  { font-size: 1.25rem; font-weight: 800; margin-top: 0.3rem; }
.author-card .sub   { font-size: 0.78rem; opacity: 0.92; margin-top: 0.15rem; }

/* Hero */
.hero-card {
    background: linear-gradient(120deg, #d6e9fb 0%, #e4f1fc 45%, #cfe6f9 100%);
    border: 1px solid #cfe2f4; border-radius: 22px;
    padding: 1.8rem 2.2rem; margin-bottom: 1.4rem;
    box-shadow: 0 6px 24px rgba(91,155,213,0.12);
}
.hero-card h1 { margin: 0; font-size: 1.9rem; color: #2f4a66; font-weight: 800; }
.hero-card p  { color: #5a7798; margin: 0.4rem 0 0 0; font-size: 0.95rem; }

/* KPI cards */
.kpi-card {
    background: #ffffff; border: 1px solid #e2ecf5; border-radius: 16px;
    padding: 1.3rem 1rem; text-align: center;
    box-shadow: 0 3px 14px rgba(91,155,213,0.08);
}
.kpi-value { font-size: 2rem; font-weight: 800; color: #2f4a66; line-height: 1.1; }
.kpi-label { font-size: 0.72rem; color: #93a3b5; text-transform: uppercase;
             letter-spacing: 0.09em; margin-top: 0.3rem; }

/* Résultat */
.result-danger {
    background: linear-gradient(135deg, #fdeced 0%, #fce4ef 100%);
    border: 2px solid #f3b4b4; border-radius: 20px;
    padding: 1.6rem; text-align: center; margin: 0.5rem 0;
}
.result-safe {
    background: linear-gradient(135deg, #e7f7ef 0%, #e4f4f6 100%);
    border: 2px solid #aee0c8; border-radius: 20px;
    padding: 1.6rem; text-align: center; margin: 0.5rem 0;
}
.result-proba { font-size: 3rem; font-weight: 800; line-height: 1.1; }
.result-sub   { color: #8b91a8; font-size: 0.9rem; margin-top: 0.3rem; }

/* Flags features */
.feat-risk { background: #fdecec; border-left: 4px solid #e88a8a; border-radius: 8px;
             padding: 7px 13px; margin: 4px 0; font-size: 0.87rem; color: #c0656a; }
.feat-ok   { background: #e8f6ef; border-left: 4px solid #6cc8a0; border-radius: 8px;
             padding: 7px 13px; margin: 4px 0; font-size: 0.87rem; color: #3f9573; }

/* Badges */
.badge-on  { background:#e3f6ec; color:#3f9573; border-radius:20px; padding:3px 13px; font-size:0.78rem; font-weight:700; }
.badge-off { background:#fdecec; color:#c0656a; border-radius:20px; padding:3px 13px; font-size:0.78rem; font-weight:700; }

/* Rows */
.soft-row { background:#ffffff; border:1px solid #e2ecf5; border-radius:12px;
            padding:0.75rem 1.2rem; margin:0.45rem 0; display:flex; align-items:center;
            gap:1.1rem; box-shadow: 0 2px 10px rgba(91,155,213,0.06); }

/* Form & buttons */
div[data-testid="stForm"] { background:#ffffff; border:1px solid #e2ecf5; border-radius:18px;
    padding:1.6rem; box-shadow:0 4px 18px rgba(91,155,213,0.08); }
[data-testid="stForm"] button[kind="formSubmit"], .stButton button {
    background: linear-gradient(120deg,#5b9bd5 0%, #7ec0ef 100%);
    color:#ffffff; border:none; border-radius:12px; font-weight:700; }
.stButton button:hover { filter: brightness(1.05); }

/* Boutons-liens */
.link-btn { display:inline-block; background:linear-gradient(120deg,#5b9bd5,#7ec0ef);
    color:#ffffff !important; padding:9px 18px; border-radius:11px; font-size:0.83rem;
    font-weight:700; text-decoration:none !important; margin:4px 8px 4px 0;
    box-shadow:0 3px 10px rgba(91,155,213,0.25); transition:filter .15s; }
.link-btn:hover { filter:brightness(1.06); }
.link-btn.ghost { background:#ffffff; color:#5b9bd5 !important;
    border:1.5px solid #cfe2f4; box-shadow:none; }
.link-btn.gh { background:linear-gradient(120deg,#2f3a4a,#44506a); }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

# ─── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers API
# ═══════════════════════════════════════════════════════════════════════════════
def get_api_health() -> tuple[bool, dict]:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        r.raise_for_status()
        return True, r.json()
    except Exception:
        return False, {}


def get_model_info() -> dict:
    try:
        r = httpx.get(f"{API_URL}/model-info", timeout=3.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def make_gauge(proba: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-0.25, 1.35)
    ax.set_aspect("equal"); ax.axis("off")

    def arc_band(a1, a2, r_in, r_out, color):
        theta = np.linspace(a1, a2, 150)
        xo, yo = r_out * np.cos(theta), r_out * np.sin(theta)
        xi, yi = r_in * np.cos(theta), r_in * np.sin(theta)
        ax.fill(np.concatenate([xo, xi[::-1]]), np.concatenate([yo, yi[::-1]]), color=color)

    arc_band(0, np.pi, 0.55, 1.0, "#eef0f8")
    arc_band(np.pi * 0.6, np.pi, 0.57, 0.98, "#a8ddc0")
    arc_band(np.pi * 0.3, np.pi * 0.6, 0.57, 0.98, "#f5d6a8")
    arc_band(0, np.pi * 0.3, 0.57, 0.98, "#f3b0b0")

    for pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        angle = np.pi * (1 - pct)
        ax.plot([np.cos(angle), 1.11 * np.cos(angle)],
                [np.sin(angle), 1.11 * np.sin(angle)], color="#c4c9dc", lw=1.5)
        ax.text(1.25 * np.cos(angle), 1.25 * np.sin(angle), f"{pct:.0%}",
                ha="center", va="center", fontsize=7.5, color=MUTED)

    needle = np.pi * (1 - proba)
    ax.annotate("", xy=(0.86 * np.cos(needle), 0.86 * np.sin(needle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#4a4f6a", lw=2.4, mutation_scale=18))
    ax.add_patch(plt.Circle((0, 0), 0.065, color="#4a4f6a", zorder=5))

    ax.text(0, 0.32, f"{proba:.1%}", ha="center", va="center",
            fontsize=30, fontweight="bold", color=CORAL if proba >= 0.5 else MINT)
    ax.text(0, 0.16, "probabilité de risque", ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(-1.38, -0.12, "Faible", ha="center", fontsize=9, color="#3f9573", fontweight="bold")
    ax.text(1.38, -0.12, "Élevé", ha="center", fontsize=9, color="#c0656a", fontweight="bold")
    fig.tight_layout(pad=0.3)
    return fig


def flag_features(p: dict) -> list[tuple[str, str, bool]]:
    return [
        ("Âge", str(int(p["age"])), p["age"] > 55),
        ("Pression (trestbps)", str(int(p["trestbps"])), p["trestbps"] > 140),
        ("Cholestérol", str(int(p["chol"])), p["chol"] > 240),
        ("FC max (thalach)", str(int(p["thalach"])), p["thalach"] < 120),
        ("Dépression ST", str(p["oldpeak"]), p["oldpeak"] > 2.0),
        ("Vaisseaux (ca)", str(int(p["ca"])), p["ca"] > 0),
        ("Angine (exang)", "Oui" if p["exang"] else "Non", p["exang"] == 1),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers MLflow
# ═══════════════════════════════════════════════════════════════════════════════
def mlflow_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{MLFLOW_URL}{path}", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_mlflow_health() -> tuple[bool, str]:
    if mlflow_get("/health") is not None:
        return True, "En ligne"
    try:
        r = httpx.get(MLFLOW_URL, timeout=4.0)
        return r.status_code < 500, "En ligne"
    except Exception:
        return False, "Hors ligne"


def get_experiment_id(name: str) -> str | None:
    data = mlflow_get(f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}")
    if data and "experiment" in data:
        return data["experiment"]["experiment_id"]
    return None


def search_runs(experiment_id: str, n: int = 200) -> list[dict]:
    try:
        r = httpx.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={"experiment_ids": [experiment_id], "max_results": n,
                  "order_by": ["attributes.start_time DESC"]},
            timeout=6.0,
        )
        r.raise_for_status()
        return r.json().get("runs", [])
    except Exception:
        return []


def get_model_comparison(experiment_id: str) -> dict[str, dict]:
    """Dernières métriques par modèle (random_forest / xgboost / lightgbm / logreg)."""
    targets = {"random_forest", "xgboost", "lightgbm", "logreg"}
    out: dict[str, dict] = {}
    for run in search_runs(experiment_id):
        name = (run["info"].get("run_name") or "").split("-")[0]
        key = run["info"].get("run_name") or ""
        label = key if key in targets else (name if name in targets else None)
        if label and label not in out:
            metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
            if any(k in metrics for k in ("roc_auc", "f1")):
                out[label] = metrics
    return out


def get_registered_models() -> list[dict]:
    data = mlflow_get("/api/2.0/mlflow/registered-models/list")
    return data.get("registered_models", []) if data else []


def get_model_versions(name: str) -> list[dict]:
    data = mlflow_get(f"/api/2.0/mlflow/model-versions/search?filter=name%3D%27{name}%27")
    return data.get("model_versions", []) if data else []


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers Airflow
# ═══════════════════════════════════════════════════════════════════════════════
def airflow_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{AIRFLOW_URL}{path}", auth=(AIRFLOW_USER, AIRFLOW_PASSWORD), timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_airflow_health() -> bool:
    data = airflow_get("/health")
    return bool(data) and data.get("metadatabase", {}).get("status") == "healthy"


def get_dag_runs(dag_id: str, n: int = 10) -> list[dict]:
    data = airflow_get(f"/api/v1/dags/{dag_id}/dagRuns?order_by=-execution_date&limit={n}")
    return data.get("dag_runs", []) if data else []


def _fmt_dt(value):
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return value


def _duration(start, end):
    if not start or not end:
        return "-"
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        secs = int((e - s).total_seconds())
        return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"
    except Exception:
        return "-"


# ═══════════════════════════════════════════════════════════════════════════════
# Évaluation : prédictions sur le jeu de test via l'API
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def evaluate_on_testset() -> dict | None:
    try:
        from heart.config import TARGET
        from heart.data import get_splits, load_data
    except Exception:
        return None
    try:
        df = load_data()
        _, x_test, _, y_test = get_splits(df)
        y_true, preds, probas = list(map(int, y_test)), [], []
        with httpx.Client(base_url=API_URL, timeout=15.0) as c:
            c.get("/health").raise_for_status()
            for _, row in x_test.iterrows():
                resp = c.post("/predict", json=json.loads(row.to_json()))
                resp.raise_for_status()
                j = resp.json()
                preds.append(int(j["label"]))
                probas.append(float(j["probability"]))
        return {"y_true": y_true, "y_pred": preds, "y_proba": probas}
    except Exception as exc:
        return {"error": str(exc)}


def kpi(col, value, label, color=INK):
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar : auteur + navigation
# ═══════════════════════════════════════════════════════════════════════════════
api_ok, _ = get_api_health()

with st.sidebar:
    st.markdown(f"""
    <div class="author-card">
        <div class="title">🫀 Heart Disease Classifier</div>
        <div class="name">{AUTHOR}</div>
        <div class="sub">Projet MLOps · ESGI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<a class="link-btn gh" href="{GITHUB_URL}" target="_blank" '
        f'style="display:block; text-align:center; margin-bottom:0.8rem;">'
        f'📦 Code source (GitHub)</a>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["📖  Contexte métier", "🔍  Prédiction", "🧪  Comparaison des modèles",
         "🎯  Évaluation du modèle", "📋  Historique", "📊  Statistiques",
         "🔧  Infrastructure"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    badge = '<span class="badge-on">● En ligne</span>' if api_ok \
            else '<span class="badge-off">● Hors ligne</span>'
    st.markdown(f"**Statut API** {badge}", unsafe_allow_html=True)

    info = get_model_info() if api_ok else {}
    if info:
        st.markdown(f"**Modèle** : `{info.get('model_name', '-')}`")
        st.markdown(f"Alias : `{info.get('model_alias', '-')}`")

    if st.session_state.history:
        total_s = len(st.session_state.history)
        risk_s = sum(1 for h in st.session_state.history if h["label"] == 1)
        st.markdown("---")
        st.markdown(f"**Session** : {total_s} préd. · {risk_s} à risque")
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-card" style="text-align:center;">
  <h1>🫀 Heart Disease Classifier</h1>
  <p>Estimation du risque de maladie cardiaque à partir de données cliniques —
     par <b>{AUTHOR}</b></p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Contexte métier
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📖  Contexte métier":
    st.markdown("### 📖 Contexte métier & problématique")
    st.markdown("""
<div class="kpi-card" style="text-align:left; padding:1.6rem 2rem;">
  <h4 style="margin-top:0;">🫀 Le problème</h4>
  <p style="color:#5a6078; font-size:0.95rem; line-height:1.6;">
    Les <b>maladies cardiovasculaires</b> sont la première cause de mortalité dans le monde
    (~18 millions de décès/an, OMS). Un dépistage <b>précoce</b> du risque permet d'orienter
    les patients vers des examens approfondis et de réduire la mortalité.
  </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="kpi-card" style="text-align:left; padding:1.4rem 1.6rem; height:100%;">
  <h4 style="margin-top:0;">🎯 L'objectif</h4>
  <p style="color:#5a6078; font-size:0.92rem; line-height:1.55;">
    Un <b>outil d'aide à la décision</b> estimant, à partir de 13 indicateurs cliniques,
    la <b>probabilité de maladie cardiaque</b>. Il ne remplace pas le médecin : il priorise et alerte.
  </p>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="kpi-card" style="text-align:left; padding:1.4rem 1.6rem; height:100%;">
  <h4 style="margin-top:0;">📊 Les données</h4>
  <p style="color:#5a6078; font-size:0.92rem; line-height:1.55;">
    <b>Heart Disease UCI (Cleveland)</b> : 303 patients, 13 variables cliniques + cible binaire
    (<b>0 = sain</b>, <b>1 = maladie</b>).
  </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ La chaîne MLOps")
    for icon, title, desc in [
        ("📥", "Données", "Préparation du jeu Heart Disease UCI."),
        ("🧠", "Entraînement", "LogReg, Random Forest, XGBoost, LightGBM + Optuna."),
        ("📊", "MLflow", "Suivi des métriques (F1, ROC AUC) et versionnage des modèles."),
        ("🤖", "API", "Le meilleur modèle est servi via FastAPI (/predict)."),
        ("🫀", "Frontend", "Cette interface envoie les données patient et affiche le risque."),
        ("🌀", "Airflow", "Ré-entraînement orchestré et planifié automatiquement."),
    ]:
        st.markdown(f"""<div class="soft-row"><span style="font-size:1.4rem;">{icon}</span>
            <span style="color:#3a3f55; font-weight:700; min-width:120px;">{title}</span>
            <span style="color:#6b7280; font-size:0.88rem;">{desc}</span></div>""",
            unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⚠️ Projet pédagogique — ne constitue pas un dispositif médical.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Prédiction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Prédiction":
    col_form, col_res = st.columns([1, 1], gap="large")
    with col_form:
        with st.form("prediction_form"):
            st.markdown("### 👤 Informations patient")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Âge (années)", min_value=1, max_value=120, value=52)
                trestbps = st.number_input("Pression artérielle (mm Hg)", value=120)
                chol = st.number_input("Cholestérol (mg/dl)", value=240)
                thalach = st.number_input("FC max atteinte", value=150)
                oldpeak = st.number_input("Dépression ST", value=1.0, step=0.1)
                ca = st.selectbox("Vaisseaux colorés (0-3)", [0, 1, 2, 3])
            with c2:
                sex = st.selectbox("Sexe", [1, 0], format_func=lambda x: "Homme" if x else "Femme")
                cp = st.selectbox("Douleur thoracique (1-4)", [1, 2, 3, 4])
                fbs = st.selectbox("Glycémie > 120 mg/dl", [0, 1], format_func=lambda x: "Oui" if x else "Non")
                restecg = st.selectbox("ECG au repos (0-2)", [0, 1, 2])
                exang = st.selectbox("Angine à l'effort", [0, 1], format_func=lambda x: "Oui" if x else "Non")
                slope = st.selectbox("Pente segment ST (1-3)", [1, 2, 3])
                thal = st.selectbox("Thalassémie", [3, 6, 7],
                                    format_func=lambda x: {3: "Normal", 6: "Défaut fixe", 7: "Défaut réversible"}[x])
            submitted = st.form_submit_button("🔍  Lancer la prédiction", use_container_width=True)

    with col_res:
        st.markdown("### 📈 Résultat")
        if not st.session_state.history and not submitted:
            st.markdown("""<div style="text-align:center;color:#b3b8cc;padding:3rem 1rem;">
                <div style="font-size:3.5rem;">🫀</div>
                <p style="margin-top:1rem;">Remplissez le formulaire et lancez une prédiction.</p></div>""",
                unsafe_allow_html=True)
        if submitted:
            if not api_ok:
                st.error("API indisponible — impossible de lancer la prédiction.")
            else:
                payload = dict(age=age, trestbps=trestbps, chol=chol, thalach=thalach,
                               oldpeak=oldpeak, ca=ca, sex=sex, cp=cp, fbs=fbs,
                               restecg=restecg, exang=exang, slope=slope, thal=thal)
                try:
                    resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
                    resp.raise_for_status()
                    result = resp.json()
                    label, proba = int(result["label"]), float(result["probability"])
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "label": label, "probability": proba, **payload})
                    fig = make_gauge(proba)
                    st.pyplot(fig, use_container_width=True); plt.close(fig)
                    if label == 1:
                        st.markdown(f"""<div class="result-danger">
                            <div style="font-size:1.3rem;font-weight:700;color:#c0656a;">⚠️ Risque élevé détecté</div>
                            <div class="result-proba" style="color:#e88a8a;">{proba:.1%}</div>
                            <div class="result-sub">probabilité de maladie cardiaque</div></div>""",
                            unsafe_allow_html=True)
                        st.error("Orienter vers une évaluation cardiologique approfondie.")
                    else:
                        st.markdown(f"""<div class="result-safe">
                            <div style="font-size:1.3rem;font-weight:700;color:#3f9573;">✅ Risque faible</div>
                            <div class="result-proba" style="color:#6cc8a0;">{proba:.1%}</div>
                            <div class="result-sub">probabilité de maladie cardiaque</div></div>""",
                            unsafe_allow_html=True)
                        st.success("Aucun signe clinique préoccupant. Suivi de routine recommandé.")
                    st.markdown("#### 🔎 Analyse des facteurs")
                    for name, val, is_risk in flag_features(payload):
                        css = "feat-risk" if is_risk else "feat-ok"
                        icon = "⚠️" if is_risk else "✓"
                        st.markdown(f'<div class="{css}">{icon} <b>{name}</b> : {val}</div>',
                                    unsafe_allow_html=True)
                except Exception as exc:
                    st.error(f"Erreur API : {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Comparaison des modèles
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Comparaison des modèles":
    st.markdown("### 🧪 Comparaison des modèles (MLflow)")
    st.caption("Métriques des modèles entraînés et comparés sur le jeu Heart Disease UCI.")

    mlf_ok, _ = get_mlflow_health()
    exp_id = get_experiment_id(EXPERIMENT_NAME) if mlf_ok else None
    comparison = get_model_comparison(exp_id) if exp_id else {}

    pretty = {"random_forest": "🌲 Random Forest", "xgboost": "⚡ XGBoost",
              "lightgbm": "💡 LightGBM", "logreg": "📈 Logistic Regression"}

    if not comparison:
        st.warning("Aucune métrique de comparaison trouvée dans MLflow.")
    else:
        rows = []
        for key, m in comparison.items():
            rows.append({
                "Modèle": pretty.get(key, key),
                "ROC AUC": m.get("roc_auc"),
                "F1": m.get("f1"),
                "CV ROC AUC": m.get("cv_roc_auc"),
            })
        df_cmp = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False, na_position="last")

        best = df_cmp.iloc[0]
        k1, k2, k3 = st.columns(3)
        kpi(k1, best["Modèle"].split(" ", 1)[-1], "Meilleur modèle", BLUE)
        kpi(k2, f"{best['ROC AUC']:.3f}" if pd.notna(best["ROC AUC"]) else "-", "Meilleur ROC AUC", MINT)
        kpi(k3, len(df_cmp), "Modèles comparés", INK)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown("#### 📊 ROC AUC par modèle")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_facecolor("#faf9ff")
            d = df_cmp.dropna(subset=["ROC AUC"])
            colors = [BLUE, MINT, AMBER, SKY][:len(d)]
            bars = ax.barh(d["Modèle"], d["ROC AUC"], color=colors, edgecolor="none")
            ax.set_xlim(0, 1.0)
            for b, v in zip(bars, d["ROC AUC"]):
                ax.text(v - 0.04, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                        va="center", ha="right", color="#ffffff", fontweight="bold", fontsize=10)
            ax.set_xlabel("ROC AUC")
            for s in ax.spines.values():
                s.set_edgecolor("#e5e7f0")
            ax.invert_yaxis()
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c2:
            st.markdown("#### 📋 Tableau récapitulatif")
            disp = df_cmp.copy()
            for col in ("ROC AUC", "F1", "CV ROC AUC"):
                disp[col] = disp[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
            st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown(
            f"<a class='link-btn' href='{MLFLOW_PUBLIC_URL}' target='_blank'>"
            f"📊 Voir le détail dans MLflow</a>",
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Évaluation du modèle
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Évaluation du modèle":
    st.markdown("### 🎯 Évaluation du modèle servi par l'API")
    st.caption("Prédictions du modèle en production sur le jeu de test (Heart Disease UCI).")

    if not api_ok:
        st.error("API indisponible — impossible d'évaluer le modèle.")
    else:
        with st.spinner("Évaluation sur le jeu de test via l'API…"):
            ev = evaluate_on_testset()

        if ev is None:
            st.warning("Jeu de données indisponible côté frontend (montage `data/` manquant).")
        elif "error" in ev:
            st.error(f"Erreur pendant l'évaluation : {ev['error']}")
        else:
            from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                         precision_score, recall_score, roc_auc_score, roc_curve)
            y_true, y_pred, y_proba = ev["y_true"], ev["y_pred"], ev["y_proba"]

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc_v = roc_auc_score(y_true, y_proba)
            except Exception:
                auc_v = float("nan")

            k1, k2, k3, k4, k5 = st.columns(5)
            kpi(k1, f"{acc:.1%}", "Exactitude", BLUE)
            kpi(k2, f"{prec:.1%}", "Précision", MINT)
            kpi(k3, f"{rec:.1%}", "Rappel", AMBER)
            kpi(k4, f"{f1:.3f}", "F1-score", SKY)
            kpi(k5, f"{auc_v:.3f}" if auc_v == auc_v else "-", "ROC AUC", CORAL)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("#### 🧮 Matrice de confusion")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4.2))
                im = ax.imshow(cm, cmap="Blues")
                labels = ["Sain (0)", "Maladie (1)"]
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(labels); ax.set_yticklabels(labels)
                ax.set_xlabel("Prédiction"); ax.set_ylabel("Réel")
                thresh = cm.max() / 2
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > thresh else INK,
                                fontsize=18, fontweight="bold")
                for s in ax.spines.values():
                    s.set_visible(False)
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

            with c2:
                st.markdown("#### 📈 Courbe ROC")
                fig, ax = plt.subplots(figsize=(5, 4.2))
                ax.set_facecolor("#faf9ff")
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"AUC = {auc_v:.3f}")
                ax.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
                ax.plot([0, 1], [0, 1], "--", color="#c4c9dc", lw=1.2)
                ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
                ax.legend(loc="lower right", frameon=False)
                for s in ax.spines.values():
                    s.set_edgecolor("#e5e7f0")
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

            st.caption(f"Évaluation sur {len(y_true)} patients du jeu de test.")

            # SHAP (importance des variables) si disponible
            shap_candidates = ["/app/models/shap_summary.png", "models/shap_summary.png"]
            shap_path = next((p for p in shap_candidates if os.path.exists(p)), None)
            if shap_path:
                st.markdown("#### 🔬 Importance des variables (SHAP)")
                st.image(shap_path, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Historique
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Historique":
    st.markdown("### 📋 Historique de la session")
    if not st.session_state.history:
        st.info("Aucune prédiction effectuée dans cette session.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        df_disp = df_hist[["timestamp", "probability", "label", "age", "chol", "trestbps", "thalach"]].copy()
        df_disp["label"] = df_disp["label"].map({0: "✅ Sain", 1: "⚠️ Risque"})
        df_disp["probability"] = df_disp["probability"].apply(lambda x: f"{x:.1%}")
        df_disp.columns = ["Heure", "Probabilité", "Résultat", "Âge", "Cholestérol", "Pression", "FC max"]
        st.dataframe(df_disp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Statistiques
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Statistiques":
    st.markdown("### 📊 Statistiques de la session")
    if not st.session_state.history:
        st.info("Lancez des prédictions pour voir les statistiques apparaître ici.")
    else:
        df_s = pd.DataFrame(st.session_state.history)
        total = len(df_s); at_risk = int((df_s["label"] == 1).sum())
        safe = total - at_risk; avg_p = float(df_s["probability"].mean())
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, total, "Total patients", INK)
        kpi(k2, at_risk, "À risque", CORAL)
        kpi(k3, safe, "Sains", MINT)
        kpi(k4, f"{avg_p:.0%}", "Risque moyen", BLUE)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Répartition des résultats")
            fig, ax = plt.subplots(figsize=(5, 4))
            if at_risk == 0:
                sizes, lbl, col = [safe], ["Sains"], [MINT]
            elif safe == 0:
                sizes, lbl, col = [at_risk], ["À risque"], [CORAL]
            else:
                sizes, lbl, col = [at_risk, safe], ["À risque", "Sains"], [CORAL, MINT]
            _, _, autotexts = ax.pie(sizes, labels=lbl, colors=col, autopct="%1.0f%%",
                                     startangle=90, textprops={"color": "#ffffff", "fontsize": 11},
                                     wedgeprops={"edgecolor": "#ffffff", "linewidth": 2})
            for at in autotexts:
                at.set_fontweight("bold")
            ax.set_title("Répartition", color=INK, fontsize=13, pad=12)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c2:
            st.markdown("#### Probabilité par patient")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.set_facecolor("#faf9ff")
            probas = df_s["probability"].values
            ax.bar(range(len(probas)), probas,
                   color=[CORAL if p >= 0.5 else MINT for p in probas], edgecolor="none")
            ax.axhline(0.5, color="#c4c9dc", linestyle="--", lw=1.2)
            ax.set_ylim(0, 1); ax.set_xlabel("Patient #"); ax.set_ylabel("Probabilité")
            for s in ax.spines.values():
                s.set_edgecolor("#e5e7f0")
            ax.set_title("Évolution des probabilités", color=INK, fontsize=13, pad=12)
            st.pyplot(fig, use_container_width=True); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE — Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧  Infrastructure":
    st.markdown("### 🔧 État des services")
    mlf_ok, _ = get_mlflow_health()
    airflow_ok = get_airflow_health()
    info2 = get_model_info() if api_ok else {}

    frontend_url = API_PUBLIC_URL.replace(":8000", ":8502").replace("/docs", "")
    st.markdown("##### 🔗 Accès rapide")
    st.markdown(f"""
        <a class="link-btn" href="{API_PUBLIC_URL}/docs" target="_blank">🤖 API (Swagger)</a>
        <a class="link-btn" href="{MLFLOW_PUBLIC_URL}" target="_blank">📊 MLflow</a>
        <a class="link-btn" href="{AIRFLOW_PUBLIC_URL}" target="_blank">🌀 Airflow</a>
        <a class="link-btn" href="{frontend_url}" target="_blank">🫀 Frontend</a>
        <a class="link-btn gh" href="{GITHUB_URL}" target="_blank">📦 GitHub</a>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    def svc_card(col, icon, name, ok, detail, url):
        with col:
            color = MINT if ok else CORAL
            status = "● En ligne" if ok else "● Hors ligne"
            st.markdown(f"""<div class="kpi-card" style="text-align:left; padding:1.2rem 1.4rem;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-weight:700; font-size:1rem; margin:0.4rem 0 0.2rem; color:#3a3f55;">{name}</div>
                <div style="color:{color}; font-size:0.82rem; font-weight:700;">{status}</div>
                <div style="color:#9aa0b5; font-size:0.78rem; margin-top:0.3rem;">{detail}</div>
                <div style="margin-top:0.7rem;"><a class="link-btn" href="{url}"
                   target="_blank" style="padding:6px 14px; font-size:0.76rem;">Ouvrir ↗</a></div>
                </div>""", unsafe_allow_html=True)

    svc_card(s1, "🤖", "API FastAPI", api_ok, f"modèle : {info2.get('model_name','-')}", f"{API_PUBLIC_URL}/docs")
    svc_card(s2, "📊", "MLflow", mlf_ok, "expériences & registry", MLFLOW_PUBLIC_URL)
    svc_card(s3, "🫀", "Frontend", True, "cette interface",
             API_PUBLIC_URL.replace(":8000", ":8502").replace("/docs", ""))
    svc_card(s4, "🌀", "Airflow", airflow_ok, "orchestration", AIRFLOW_PUBLIC_URL)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤖 API FastAPI — Endpoints")
    for method, path_, desc in [
        ("GET", "/health", "Vérifie que l'API et le modèle sont chargés"),
        ("POST", "/predict", "Prédit la probabilité de maladie cardiaque"),
        ("GET", "/model-info", "Retourne le nom, alias et source du modèle"),
        ("GET", "/docs", "Documentation Swagger interactive"),
    ]:
        m_color = MINT if method == "GET" else SKY
        st.markdown(f"""<div class="soft-row">
            <span style="background:{m_color}22; color:{m_color}; font-weight:700; font-size:0.78rem;
                         padding:3px 10px; border-radius:6px; min-width:48px; text-align:center;">{method}</span>
            <span style="color:#3a3f55; font-family:monospace; font-size:0.9rem;">{API_PUBLIC_URL}{path_}</span>
            <span style="color:#9aa0b5; font-size:0.85rem; margin-left:auto;">{desc}</span></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌀 Airflow — Historique des ré-entraînements")
    st.caption(f"DAG `{RETRAIN_DAG_ID}` · prépare les données → entraîne → contrôle qualité")
    if not airflow_ok:
        st.warning("Airflow inaccessible depuis le frontend.")
    else:
        runs = get_dag_runs(RETRAIN_DAG_ID, n=10)
        if not runs:
            st.info("Aucun ré-entraînement enregistré pour le moment.")
        else:
            state_badge = {"success": (MINT, "✅ Réussi"), "running": (SKY, "🔄 En cours"),
                           "failed": (CORAL, "❌ Échoué"), "queued": (AMBER, "⏳ En file")}
            n_ok = sum(1 for r in runs if r.get("state") == "success")
            k1, k2 = st.columns(2)
            kpi(k1, len(runs), "Runs affichés", INK)
            kpi(k2, n_ok, "Réussis", MINT)
            st.markdown("<br>", unsafe_allow_html=True)
            for r in runs:
                color, label = state_badge.get(r.get("state", ""), (MUTED, r.get("state") or "-"))
                trigger = (r.get("run_type") or "").replace("_", " ")
                st.markdown(f"""<div class="soft-row" style="border-left:4px solid {color};">
                    <span style="color:{color}; font-weight:700; font-size:0.85rem; min-width:90px;">{label}</span>
                    <span style="color:#3a3f55; font-size:0.88rem;">🕒 {_fmt_dt(r.get('start_date'))}</span>
                    <span style="color:#8b91a8; font-size:0.82rem;">⏱️ {_duration(r.get('start_date'), r.get('end_date'))}</span>
                    <span style="color:#9aa0b5; font-size:0.78rem; margin-left:auto;">{trigger}</span></div>""",
                    unsafe_allow_html=True)
            st.markdown(
                f"<div style='margin-top:0.7rem;'><a class='link-btn' "
                f"href='{AIRFLOW_PUBLIC_URL}/dags/{RETRAIN_DAG_ID}/grid' target='_blank'>"
                f"🌀 Ouvrir le DAG dans Airflow</a></div>",
                unsafe_allow_html=True)


st.markdown(
    f"<hr style='border-color:#e2ecf5; margin-top:2rem;'>"
    f"<p style='text-align:center; color:#b3b8cc; font-size:0.78rem;'>"
    f"{AUTHOR} · API : {API_PUBLIC_URL} | MLflow : {MLFLOW_PUBLIC_URL} | Airflow : {AIRFLOW_PUBLIC_URL}</p>",
    unsafe_allow_html=True,
)
