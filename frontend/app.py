"""Frontend Streamlit — Heart Disease Classifier (thème clair pastel)."""
from __future__ import annotations

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

# ─── Palette pastel ──────────────────────────────────────────────────────────────
INK      = "#3a3f55"   # texte principal
MUTED    = "#8b91a8"   # texte secondaire
LAVENDER = "#8b7fd4"   # accent principal (violet pastel)
MINT     = "#6cc8a0"   # vert menthe (sain / succès)
CORAL    = "#e88a8a"   # corail (risque / échec)
AMBER    = "#e9b872"   # ambre (intermédiaire)
SKY      = "#7eb6e8"   # bleu pastel
CARD_BD  = "#ececf5"   # bordure carte

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
[data-testid="stAppViewContainer"] { background: #f5f6fc; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #efeafd 0%, #fdeaf1 100%);
    border-right: 1px solid #ececf5;
}
[data-testid="stSidebar"] * { color: #4a4f6a; }

h1, h2, h3, h4 { color: #3a3f55; }

/* Hero */
.hero-card {
    background: linear-gradient(120deg, #e8e3fb 0%, #f3e6f5 45%, #e2eefb 100%);
    border: 1px solid #e3ddf6;
    border-radius: 22px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 6px 24px rgba(139,127,212,0.10);
}
.hero-card h1 { margin: 0; font-size: 2rem; color: #4a4566; font-weight: 800; }
.hero-card p  { color: #7a7298; margin: 0.5rem 0 0 0; font-size: 0.98rem; }

/* KPI cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid #ececf5;
    border-radius: 16px;
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 3px 14px rgba(140,130,200,0.07);
}
.kpi-value { font-size: 2.1rem; font-weight: 800; color: #3a3f55; line-height: 1.1; }
.kpi-label { font-size: 0.72rem; color: #9aa0b5; text-transform: uppercase;
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
.feat-risk {
    background: #fdecec; border-left: 4px solid #e88a8a;
    border-radius: 8px; padding: 7px 13px; margin: 4px 0;
    font-size: 0.87rem; color: #c0656a;
}
.feat-ok {
    background: #e8f6ef; border-left: 4px solid #6cc8a0;
    border-radius: 8px; padding: 7px 13px; margin: 4px 0;
    font-size: 0.87rem; color: #3f9573;
}

/* Badges */
.badge-on  { background:#e3f6ec; color:#3f9573; border-radius:20px; padding:3px 13px; font-size:0.78rem; font-weight:700; }
.badge-off { background:#fdecec; color:#c0656a; border-radius:20px; padding:3px 13px; font-size:0.78rem; font-weight:700; }

/* Service / endpoint / run rows */
.soft-row {
    background:#ffffff; border:1px solid #ececf5; border-radius:12px;
    padding:0.75rem 1.2rem; margin:0.45rem 0; display:flex;
    align-items:center; gap:1.1rem;
    box-shadow: 0 2px 10px rgba(140,130,200,0.05);
}

/* Form */
div[data-testid="stForm"] {
    background: #ffffff; border: 1px solid #ececf5; border-radius: 18px;
    padding: 1.6rem; box-shadow: 0 4px 18px rgba(140,130,200,0.07);
}

/* Tabs */
button[data-baseweb="tab"] { font-size: 0.95rem; }
[data-testid="stForm"] button[kind="formSubmit"],
.stButton button {
    background: linear-gradient(120deg,#8b7fd4 0%, #b39ae0 100%);
    color: #ffffff; border: none; border-radius: 12px; font-weight: 700;
}
.stButton button:hover { filter: brightness(1.05); }
</style>
""", unsafe_allow_html=True)

# Style commun des graphiques matplotlib (thème clair)
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#ffffff",
    "text.color":       INK,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
})

# ─── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ─── Helpers API ─────────────────────────────────────────────────────────────────
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
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    def arc_band(a1: float, a2: float, r_in: float, r_out: float, color: str) -> None:
        theta = np.linspace(a1, a2, 150)
        xo, yo = r_out * np.cos(theta), r_out * np.sin(theta)
        xi, yi = r_in  * np.cos(theta), r_in  * np.sin(theta)
        ax.fill(np.concatenate([xo, xi[::-1]]), np.concatenate([yo, yi[::-1]]), color=color)

    arc_band(0, np.pi, 0.55, 1.0, "#eef0f8")                  # fond
    arc_band(np.pi * 0.6, np.pi, 0.57, 0.98, "#a8ddc0")       # vert  0–40 %
    arc_band(np.pi * 0.3, np.pi * 0.6, 0.57, 0.98, "#f5d6a8") # ambre 40–70 %
    arc_band(0, np.pi * 0.3, 0.57, 0.98, "#f3b0b0")           # rouge 70–100 %

    for pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        angle = np.pi * (1 - pct)
        ax.plot([np.cos(angle), 1.11 * np.cos(angle)],
                [np.sin(angle), 1.11 * np.sin(angle)], color="#c4c9dc", lw=1.5)
        ax.text(1.25 * np.cos(angle), 1.25 * np.sin(angle), f"{pct:.0%}",
                ha="center", va="center", fontsize=7.5, color=MUTED)

    needle = np.pi * (1 - proba)
    ax.annotate("", xy=(0.86 * np.cos(needle), 0.86 * np.sin(needle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#4a4f6a", lw=2.4,
                                mutation_scale=18))
    ax.add_patch(plt.Circle((0, 0), 0.065, color="#4a4f6a", zorder=5))

    color_score = CORAL if proba >= 0.5 else MINT
    ax.text(0, 0.32, f"{proba:.1%}", ha="center", va="center",
            fontsize=30, fontweight="bold", color=color_score)
    ax.text(0, 0.16, "probabilité de risque", ha="center", va="center",
            fontsize=9, color=MUTED)
    ax.text(-1.38, -0.12, "Faible", ha="center", fontsize=9,
            color="#3f9573", fontweight="bold")
    ax.text( 1.38, -0.12, "Élevé",  ha="center", fontsize=9,
            color="#c0656a", fontweight="bold")

    fig.tight_layout(pad=0.3)
    return fig


def flag_features(p: dict) -> list[tuple[str, str, bool]]:
    return [
        ("Âge",           str(int(p["age"])),       p["age"] > 55),
        ("Pression (trestbps)", str(int(p["trestbps"])), p["trestbps"] > 140),
        ("Cholestérol",    str(int(p["chol"])),      p["chol"] > 240),
        ("FC max (thalach)", str(int(p["thalach"])), p["thalach"] < 120),
        ("Dépression ST",  str(p["oldpeak"]),        p["oldpeak"] > 2.0),
        ("Vaisseaux (ca)", str(int(p["ca"])),        p["ca"] > 0),
        ("Angine (exang)", "Oui" if p["exang"] else "Non", p["exang"] == 1),
    ]


# ─── MLflow helpers ────────────────────────────────────────────────────────────
def mlflow_get(path: str) -> dict | None:
    try:
        r = httpx.get(f"{MLFLOW_URL}{path}", timeout=4.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_mlflow_health() -> tuple[bool, str]:
    data = mlflow_get("/health")
    if data is not None:
        return True, "En ligne"
    try:
        r = httpx.get(MLFLOW_URL, timeout=4.0)
        return r.status_code < 500, "En ligne"
    except Exception:
        return False, "Hors ligne"


# ─── Airflow helpers ─────────────────────────────────────────────────────────────
def airflow_get(path: str) -> dict | None:
    try:
        r = httpx.get(
            f"{AIRFLOW_URL}{path}",
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            timeout=5.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_airflow_health() -> bool:
    data = airflow_get("/health")
    if not data:
        return False
    return data.get("metadatabase", {}).get("status") == "healthy"


def get_dag_runs(dag_id: str, n: int = 10) -> list[dict]:
    data = airflow_get(
        f"/api/v1/dags/{dag_id}/dagRuns?order_by=-execution_date&limit={n}"
    )
    if data:
        return data.get("dag_runs", [])
    return []


def _fmt_dt(value: str | None) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime(
            "%d/%m/%Y %H:%M"
        )
    except Exception:
        return value


def _duration(start: str | None, end: str | None) -> str:
    if not start or not end:
        return "-"
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        secs = int((e - s).total_seconds())
        return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"
    except Exception:
        return "-"


def kpi(col, value, label, color=INK) -> None:
    with col:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value" style="color:{color};">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Classifier")
    api_ok, _ = get_api_health()
    badge = '<span class="badge-on">● En ligne</span>' if api_ok \
            else '<span class="badge-off">● Hors ligne</span>'
    st.markdown(f"**Statut API** {badge}", unsafe_allow_html=True)
    st.markdown("---")

    info = get_model_info() if api_ok else {}
    if info:
        st.markdown("**Modèle**")
        st.markdown(f"`{info.get('model_name', '-')}`")
        st.markdown(f"Alias : `{info.get('model_alias', '-')}`")
        src = info.get("model_source", "-") or "-"
        st.markdown(f"Source : `{'...' + src[-30:] if len(src) > 33 else src}`")
        st.markdown("---")

    if st.session_state.history:
        total_s = len(st.session_state.history)
        risk_s  = sum(1 for h in st.session_state.history if h["label"] == 1)
        st.markdown("**Session**")
        st.markdown(f"Prédictions : **{total_s}**")
        st.markdown(f"À risque : **{risk_s}** ({risk_s / total_s:.0%})")
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card" style="text-align:center;">
  <h1>🫀 Heart Disease Classifier</h1>
  <p>Estimation du risque de maladie cardiaque à partir de données cliniques ·
     Modèle ML entraîné sur Heart Disease UCI (303 patients)</p>
  <p style="margin-top:0.7rem; font-size:0.85rem; color:#9a8fc4; font-weight:600;">
     Projet MLOps · réalisé par <b>Tinhinane ISSAD</b></p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["📖  Contexte métier", "🔍  Prédiction", "📋  Historique",
     "📊  Statistiques", "🔧  Infrastructure"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — Contexte métier
# ═══════════════════════════════════════════════════════════════════════════════
with tab0:
    st.markdown("### 📖 Contexte métier & problématique")

    st.markdown("""
<div class="kpi-card" style="text-align:left; padding:1.6rem 2rem;">
  <h4 style="margin-top:0;">🫀 Le problème</h4>
  <p style="color:#5a6078; font-size:0.95rem; line-height:1.6;">
    Les <b>maladies cardiovasculaires</b> sont la première cause de mortalité dans
    le monde (environ 18 millions de décès par an selon l'OMS). Un dépistage
    <b>précoce</b> du risque permet d'orienter les patients vers des examens
    approfondis et de réduire considérablement la mortalité. Or, l'évaluation
    repose souvent sur l'expérience du praticien et sur de nombreux examens
    cliniques dont l'interprétation conjointe est complexe.
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
    Construire un <b>outil d'aide à la décision</b> qui estime, à partir de
    13 indicateurs cliniques simples (âge, tension, cholestérol, ECG…), la
    <b>probabilité qu'un patient présente une maladie cardiaque</b>.
    Il ne remplace pas le médecin : il <b>priorise</b> et <b>alerte</b>.
  </p>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="kpi-card" style="text-align:left; padding:1.4rem 1.6rem; height:100%;">
  <h4 style="margin-top:0;">📊 Les données</h4>
  <p style="color:#5a6078; font-size:0.92rem; line-height:1.55;">
    Jeu de données <b>Heart Disease UCI (Cleveland)</b> : 303 patients,
    13 variables cliniques + une cible binaire
    (<b>0 = sain</b>, <b>1 = maladie</b>). C'est un jeu de référence
    largement utilisé en apprentissage automatique médical.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Comment ça marche — la chaîne MLOps")
    steps = [
        ("📥", "Données",   "Préparation et nettoyage du jeu Heart Disease UCI."),
        ("🧠", "Entraînement", "Un modèle scikit-learn apprend à distinguer patients sains / à risque."),
        ("📊", "MLflow",    "Chaque entraînement est tracé (métriques F1, ROC AUC) et le modèle versionné."),
        ("🤖", "API",       "Le meilleur modèle est servi via une API FastAPI (/predict)."),
        ("🫀", "Frontend",  "Cette interface envoie les données patient à l'API et affiche le risque."),
        ("🌀", "Airflow",   "Le ré-entraînement est orchestré et planifié automatiquement."),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div class="soft-row">
            <span style="font-size:1.4rem;">{icon}</span>
            <span style="color:#3a3f55; font-weight:700; font-size:0.92rem;
                         min-width:120px;">{title}</span>
            <span style="color:#6b7280; font-size:0.88rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "⚠️ **Avertissement** : cet outil est un projet pédagogique. Il ne constitue "
        "en aucun cas un dispositif médical et ne doit pas servir à poser un diagnostic réel."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prédiction
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        with st.form("prediction_form"):
            st.markdown("### 👤 Informations patient")
            c1, c2 = st.columns(2)
            with c1:
                age      = st.number_input("Âge (années)", min_value=1, max_value=120, value=52)
                trestbps = st.number_input("Pression artérielle (mm Hg)", value=120)
                chol     = st.number_input("Cholestérol (mg/dl)", value=240)
                thalach  = st.number_input("FC max atteinte", value=150)
                oldpeak  = st.number_input("Dépression ST", value=1.0, step=0.1)
                ca       = st.selectbox("Vaisseaux colorés (0-3)", [0, 1, 2, 3])
            with c2:
                sex     = st.selectbox("Sexe", [1, 0],
                                       format_func=lambda x: "Homme" if x else "Femme")
                cp      = st.selectbox("Douleur thoracique (1-4)", [1, 2, 3, 4])
                fbs     = st.selectbox("Glycémie > 120 mg/dl", [0, 1],
                                       format_func=lambda x: "Oui" if x else "Non")
                restecg = st.selectbox("ECG au repos (0-2)", [0, 1, 2])
                exang   = st.selectbox("Angine à l'effort", [0, 1],
                                       format_func=lambda x: "Oui" if x else "Non")
                slope   = st.selectbox("Pente segment ST (1-3)", [1, 2, 3])
                thal    = st.selectbox("Thalassémie", [3, 6, 7],
                                       format_func=lambda x:
                                           {3: "Normal", 6: "Défaut fixe",
                                            7: "Défaut réversible"}[x])

            submitted = st.form_submit_button(
                "🔍  Lancer la prédiction", use_container_width=True
            )

    with col_res:
        st.markdown("### 📈 Résultat")

        if not st.session_state.history:
            st.markdown("""
            <div style="text-align:center;color:#b3b8cc;padding:3rem 1rem;">
                <div style="font-size:3.5rem;">🫀</div>
                <p style="margin-top:1rem;">
                    Remplissez le formulaire et lancez une prédiction.
                </p>
            </div>
            """, unsafe_allow_html=True)

        if submitted:
            if not api_ok:
                st.error("API indisponible — impossible de lancer la prédiction.")
            else:
                payload = dict(
                    age=age, trestbps=trestbps, chol=chol, thalach=thalach,
                    oldpeak=oldpeak, ca=ca, sex=sex, cp=cp, fbs=fbs,
                    restecg=restecg, exang=exang, slope=slope, thal=thal,
                )
                try:
                    resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
                    resp.raise_for_status()
                    result = resp.json()
                    label = int(result["label"])
                    proba = float(result["probability"])

                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "label": label, "probability": proba, **payload,
                    })

                    fig = make_gauge(proba)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    if label == 1:
                        st.markdown(f"""
                        <div class="result-danger">
                            <div style="font-size:1.3rem;font-weight:700;color:#c0656a;">
                                ⚠️ Risque élevé détecté
                            </div>
                            <div class="result-proba" style="color:#e88a8a;">{proba:.1%}</div>
                            <div class="result-sub">probabilité de maladie cardiaque</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("Orienter vers une évaluation cardiologique approfondie.")
                    else:
                        st.markdown(f"""
                        <div class="result-safe">
                            <div style="font-size:1.3rem;font-weight:700;color:#3f9573;">
                                ✅ Risque faible
                            </div>
                            <div class="result-proba" style="color:#6cc8a0;">{proba:.1%}</div>
                            <div class="result-sub">probabilité de maladie cardiaque</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("Aucun signe clinique préoccupant. Suivi de routine recommandé.")

                    st.markdown("#### 🔎 Analyse des facteurs")
                    for name, val, is_risk in flag_features(payload):
                        css = "feat-risk" if is_risk else "feat-ok"
                        icon = "⚠️" if is_risk else "✓"
                        st.markdown(
                            f'<div class="{css}">{icon} <b>{name}</b> : {val}</div>',
                            unsafe_allow_html=True,
                        )

                except Exception as exc:
                    st.error(f"Erreur API : {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Historique
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📋 Historique de la session")
    if not st.session_state.history:
        st.info("Aucune prédiction effectuée dans cette session.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        df_disp = df_hist[
            ["timestamp", "probability", "label", "age", "chol", "trestbps", "thalach"]
        ].copy()
        df_disp["label"]       = df_disp["label"].map({0: "✅ Sain", 1: "⚠️ Risque"})
        df_disp["probability"] = df_disp["probability"].apply(lambda x: f"{x:.1%}")
        df_disp.columns = ["Heure", "Probabilité", "Résultat", "Âge",
                           "Cholestérol", "Pression", "FC max"]
        st.dataframe(df_disp, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Statistiques
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Statistiques de la session")
    if not st.session_state.history:
        st.info("Lancez des prédictions pour voir les statistiques apparaître ici.")
    else:
        df_s    = pd.DataFrame(st.session_state.history)
        total   = len(df_s)
        at_risk = int((df_s["label"] == 1).sum())
        safe    = total - at_risk
        avg_p   = float(df_s["probability"].mean())

        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, total,          "Total patients", INK)
        kpi(k2, at_risk,        "À risque",       CORAL)
        kpi(k3, safe,           "Sains",          MINT)
        kpi(k4, f"{avg_p:.0%}", "Risque moyen",   LAVENDER)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Répartition des résultats")
            fig, ax = plt.subplots(figsize=(5, 4))
            if at_risk == 0:
                sizes, labels_pie, colors_pie = [safe], ["Sains"], [MINT]
            elif safe == 0:
                sizes, labels_pie, colors_pie = [at_risk], ["À risque"], [CORAL]
            else:
                sizes, labels_pie = [at_risk, safe], ["À risque", "Sains"]
                colors_pie = [CORAL, MINT]
            wedges, _, autotexts = ax.pie(
                sizes, labels=labels_pie, colors=colors_pie,
                autopct="%1.0f%%", startangle=90,
                textprops={"color": "#ffffff", "fontsize": 11},
                wedgeprops={"edgecolor": "#ffffff", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax.set_title("Répartition", color=INK, fontsize=13, pad=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown("#### Probabilité par patient")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.set_facecolor("#faf9ff")
            probas = df_s["probability"].values
            bar_colors = [CORAL if p >= 0.5 else MINT for p in probas]
            ax.bar(range(len(probas)), probas, color=bar_colors, edgecolor="none")
            ax.axhline(0.5, color="#c4c9dc", linestyle="--", lw=1.2)
            ax.text(len(probas) - 0.5, 0.52, "seuil 50 %",
                    color=MUTED, fontsize=8, ha="right")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Patient #", fontsize=9)
            ax.set_ylabel("Probabilité", fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#e5e7f0")
            ax.set_title("Évolution des probabilités", color=INK, fontsize=13, pad=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if len(df_s) >= 3:
            st.markdown("#### Corrélation Âge / Probabilité de risque")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.set_facecolor("#faf9ff")
            sc = ax.scatter(df_s["age"], df_s["probability"],
                            c=df_s["probability"], cmap="RdYlGn_r",
                            s=110, alpha=0.9, edgecolors="#ffffff",
                            linewidths=1.2, vmin=0, vmax=1)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Probabilité", color=MUTED, fontsize=9)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
            ax.set_xlabel("Âge")
            ax.set_ylabel("Probabilité de risque")
            for spine in ax.spines.values():
                spine.set_edgecolor("#e5e7f0")
            ax.set_title("Âge vs Risque cardiaque", color=INK, fontsize=13, pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔧 État des services")

    api_ok2, _ = get_api_health()
    mlf_ok, _  = get_mlflow_health()
    airflow_ok = get_airflow_health()
    info2      = get_model_info() if api_ok2 else {}

    s1, s2, s3, s4 = st.columns(4)

    def svc_card(col, icon, name, ok, detail, url):
        with col:
            color  = MINT if ok else CORAL
            status = "● En ligne" if ok else "● Hors ligne"
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left; padding:1.2rem 1.4rem;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-weight:700; font-size:1rem; margin:0.4rem 0 0.2rem;
                            color:#3a3f55;">
                    {name}
                </div>
                <div style="color:{color}; font-size:0.82rem; font-weight:700;">
                    {status}
                </div>
                <div style="color:#9aa0b5; font-size:0.78rem; margin-top:0.3rem;">
                    {detail}
                </div>
                <div style="margin-top:0.6rem;">
                    <a href="{url}" target="_blank"
                       style="color:#8b7fd4; font-size:0.78rem; font-weight:600;">
                        🔗 {url}
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

    svc_card(s1, "🤖", "API FastAPI", api_ok2,
             f"modèle : {info2.get('model_name','-')}", f"{API_PUBLIC_URL}/docs")
    svc_card(s2, "📊", "MLflow Tracking", mlf_ok,
             "expériences & model registry", MLFLOW_PUBLIC_URL)
    svc_card(s3, "🫀", "Frontend Streamlit", True,
             "cette interface",
             API_PUBLIC_URL.replace(":8000", ":8502").replace("/docs", ""))
    svc_card(s4, "🌀", "Airflow", airflow_ok,
             "orchestration des pipelines", AIRFLOW_PUBLIC_URL)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── API endpoints ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 API FastAPI — Endpoints")
    endpoints = [
        ("GET",  "/health",     "Vérifie que l'API et le modèle sont chargés"),
        ("POST", "/predict",    "Prédit la probabilité de maladie cardiaque"),
        ("GET",  "/model-info", "Retourne le nom, alias et source du modèle"),
        ("GET",  "/docs",       "Documentation Swagger interactive"),
    ]
    for method, path, desc in endpoints:
        m_color = MINT if method == "GET" else SKY
        st.markdown(f"""
        <div class="soft-row">
            <span style="background:{m_color}22; color:{m_color}; font-weight:700;
                         font-size:0.78rem; padding:3px 10px; border-radius:6px;
                         min-width:48px; text-align:center;">{method}</span>
            <span style="color:#3a3f55; font-family:monospace; font-size:0.9rem;">
                {API_PUBLIC_URL}{path}
            </span>
            <span style="color:#9aa0b5; font-size:0.85rem; margin-left:auto;">
                {desc}
            </span>
        </div>
        """, unsafe_allow_html=True)

    if api_ok2 and info2:
        with st.expander("📋 Détails du modèle chargé"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Nom** : `{info2.get('model_name', '-')}`")
                st.markdown(f"**Alias** : `{info2.get('model_alias', '-')}`")
                st.markdown(f"**Source** : `{info2.get('model_source', '-')}`")
            with col_b:
                feats = info2.get("features", {})
                st.markdown(f"**Features numériques** : {', '.join(feats.get('numeric', []))}")
                st.markdown(f"**Features catégorielles** : {', '.join(feats.get('categorical', []))}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Historique des ré-entraînements (Airflow) ───────────────────────────────
    st.markdown("### 🌀 Airflow — Historique des ré-entraînements")
    st.caption(
        f"DAG `{RETRAIN_DAG_ID}` · prépare les données → entraîne → contrôle qualité"
    )

    if not airflow_ok:
        st.warning("Airflow inaccessible depuis le frontend.")
    else:
        runs = get_dag_runs(RETRAIN_DAG_ID, n=10)
        if not runs:
            st.info("Aucun ré-entraînement enregistré pour le moment.")
        else:
            state_badge = {
                "success": (MINT,  "✅ Réussi"),
                "running": (SKY,   "🔄 En cours"),
                "failed":  (CORAL, "❌ Échoué"),
                "queued":  (AMBER, "⏳ En file"),
            }
            n_ok = sum(1 for r in runs if r.get("state") == "success")
            k1, k2 = st.columns(2)
            kpi(k1, len(runs), "Runs affichés", INK)
            kpi(k2, n_ok,      "Réussis",       MINT)

            st.markdown("<br>", unsafe_allow_html=True)

            for r in runs:
                state = r.get("state", "")
                color, label = state_badge.get(state, (MUTED, state or "-"))
                trigger = (r.get("run_type") or "").replace("_", " ")
                st.markdown(f"""
                <div class="soft-row" style="border-left:4px solid {color};">
                    <span style="color:{color}; font-weight:700; font-size:0.85rem;
                                 min-width:90px;">{label}</span>
                    <span style="color:#3a3f55; font-size:0.88rem;">
                        🕒 {_fmt_dt(r.get('start_date'))}
                    </span>
                    <span style="color:#8b91a8; font-size:0.82rem;">
                        ⏱️ {_duration(r.get('start_date'), r.get('end_date'))}
                    </span>
                    <span style="color:#9aa0b5; font-size:0.78rem; margin-left:auto;">
                        {trigger}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(
                f"<div style='margin-top:0.6rem;'>"
                f"<a href='{AIRFLOW_PUBLIC_URL}/dags/{RETRAIN_DAG_ID}/grid' "
                f"target='_blank' style='color:#8b7fd4; font-size:0.82rem; "
                f"font-weight:600;'>🔗 Ouvrir le DAG dans Airflow</a></div>",
                unsafe_allow_html=True,
            )


st.markdown(
    f"<hr style='border-color:#ececf5; margin-top:2rem;'>"
    f"<p style='text-align:center; color:#b3b8cc; font-size:0.78rem;'>"
    f"API : {API_PUBLIC_URL} &nbsp;|&nbsp; MLflow : {MLFLOW_PUBLIC_URL} "
    f"&nbsp;|&nbsp; Airflow : {AIRFLOW_PUBLIC_URL}</p>",
    unsafe_allow_html=True,
)
