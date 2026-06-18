"""Frontend Streamlit — Heart Disease Classifier (dashboard enrichi)."""
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
API_PUBLIC_URL    = os.getenv("API_PUBLIC_URL",    API_URL)
MLFLOW_PUBLIC_URL = os.getenv("MLFLOW_PUBLIC_URL", MLFLOW_URL)

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
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13162b 0%, #0f1117 100%);
    border-right: 1px solid #2d3561;
}

.hero-card {
    background: linear-gradient(135deg, #1a1f3c 0%, #251530 50%, #1a2538 100%);
    border: 1px solid #3a4070;
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
}
.hero-card h1 { margin: 0; font-size: 1.9rem; color: #e2e8f0; }
.hero-card p  { color: #8896b3; margin: 0.4rem 0 0 0; font-size: 0.95rem; }

.kpi-card {
    background: #1a1f3c;
    border: 1px solid #2d3561;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
.kpi-label { font-size: 0.75rem; color: #6b7fa8; text-transform: uppercase;
             letter-spacing: 0.08em; margin-top: 0.25rem; }

.result-danger {
    background: linear-gradient(135deg, #2d1515 0%, #1f0f0f 100%);
    border: 2px solid #e53e3e; border-radius: 18px;
    padding: 1.6rem; text-align: center; margin: 0.5rem 0;
}
.result-safe {
    background: linear-gradient(135deg, #0d2918 0%, #091d11 100%);
    border: 2px solid #38a169; border-radius: 18px;
    padding: 1.6rem; text-align: center; margin: 0.5rem 0;
}
.result-proba { font-size: 3rem; font-weight: 800; line-height: 1.1; }
.result-sub   { color: #a0aec0; font-size: 0.9rem; margin-top: 0.3rem; }

.feat-risk {
    background: #2d1515; border-left: 3px solid #e53e3e;
    border-radius: 6px; padding: 6px 12px; margin: 3px 0;
    font-size: 0.87rem; color: #fc8181;
}
.feat-ok {
    background: #0d2918; border-left: 3px solid #38a169;
    border-radius: 6px; padding: 6px 12px; margin: 3px 0;
    font-size: 0.87rem; color: #68d391;
}

.badge-on  { background:#1a3a28; color:#48bb78; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }
.badge-off { background:#3d1515; color:#fc8181; border-radius:20px; padding:3px 12px; font-size:0.78rem; font-weight:600; }

div[data-testid="stForm"] {
    background: #13162b; border: 1px solid #2d3561; border-radius: 16px; padding: 1.4rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ─── Helpers ───────────────────────────────────────────────────────────────────
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
    fig.patch.set_facecolor("#13162b")
    ax.set_facecolor("#13162b")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    def arc_band(a1: float, a2: float, r_in: float, r_out: float, color: str) -> None:
        theta = np.linspace(a1, a2, 150)
        xo, yo = r_out * np.cos(theta), r_out * np.sin(theta)
        xi, yi = r_in  * np.cos(theta), r_in  * np.sin(theta)
        ax.fill(np.concatenate([xo, xi[::-1]]), np.concatenate([yo, yi[::-1]]), color=color)

    arc_band(0, np.pi, 0.55, 1.0, "#1e2540")           # fond
    arc_band(np.pi * 0.6, np.pi, 0.57, 0.98, "#276749") # vert  0–40 %
    arc_band(np.pi * 0.3, np.pi * 0.6, 0.57, 0.98, "#c05621") # orange 40–70 %
    arc_band(0, np.pi * 0.3, 0.57, 0.98, "#9b2c2c")     # rouge  70–100 %

    for pct in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        angle = np.pi * (1 - pct)
        ax.plot([np.cos(angle), 1.11 * np.cos(angle)],
                [np.sin(angle), 1.11 * np.sin(angle)], color="#a0aec0", lw=1.5)
        ax.text(1.25 * np.cos(angle), 1.25 * np.sin(angle), f"{pct:.0%}",
                ha="center", va="center", fontsize=7.5, color="#a0aec0")

    needle = np.pi * (1 - proba)
    ax.annotate("", xy=(0.86 * np.cos(needle), 0.86 * np.sin(needle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2,
                                mutation_scale=18))
    ax.add_patch(plt.Circle((0, 0), 0.065, color="#e2e8f0", zorder=5))

    color_score = "#e53e3e" if proba >= 0.5 else "#38a169"
    ax.text(0, 0.32, f"{proba:.1%}", ha="center", va="center",
            fontsize=30, fontweight="bold", color=color_score)
    ax.text(0, 0.16, "probabilité de risque", ha="center", va="center",
            fontsize=9, color="#8896b3")
    ax.text(-1.38, -0.12, "Faible", ha="center", fontsize=9,
            color="#68d391", fontweight="bold")
    ax.text( 1.38, -0.12, "Élevé",  ha="center", fontsize=9,
            color="#fc8181", fontweight="bold")

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
    # try root
    try:
        r = httpx.get(MLFLOW_URL, timeout=4.0)
        return r.status_code < 500, "En ligne"
    except Exception:
        return False, "Hors ligne"


def get_experiments() -> list[dict]:
    data = mlflow_get("/api/2.0/mlflow/experiments/search?max_results=10")
    if data:
        return data.get("experiments", [])
    return []


def get_recent_runs(experiment_id: str, n: int = 8) -> list[dict]:
    try:
        r = httpx.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [experiment_id],
                "max_results": n,
                "order_by": ["attributes.start_time DESC"],
            },
            timeout=4.0,
        )
        r.raise_for_status()
        return r.json().get("runs", [])
    except Exception:
        return []


def get_registered_models() -> list[dict]:
    data = mlflow_get("/api/2.0/mlflow/registered-models/list")
    if data:
        return data.get("registered_models", [])
    return []


def get_model_versions(name: str) -> list[dict]:
    data = mlflow_get(f"/api/2.0/mlflow/model-versions/search?filter=name%3D%27{name}%27")
    if data:
        return data.get("model_versions", [])
    return []

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
<div class="hero-card">
  <h1>🫀 Heart Disease Classifier</h1>
  <p>Estimation du risque de maladie cardiaque à partir de données cliniques ·
     Modèle ML entraîné sur Heart Disease UCI (303 patients)</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍  Prédiction", "📋  Historique", "📊  Statistiques", "🔧  Infrastructure"])

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
            <div style="text-align:center;color:#4a5568;padding:3rem 1rem;">
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
                            <div style="font-size:1.3rem;font-weight:700;color:#fc8181;">
                                ⚠️ Risque élevé détecté
                            </div>
                            <div class="result-proba" style="color:#e53e3e;">{proba:.1%}</div>
                            <div class="result-sub">probabilité de maladie cardiaque</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("Orienter vers une évaluation cardiologique approfondie.")
                    else:
                        st.markdown(f"""
                        <div class="result-safe">
                            <div style="font-size:1.3rem;font-weight:700;color:#68d391;">
                                ✅ Risque faible
                            </div>
                            <div class="result-proba" style="color:#38a169;">{proba:.1%}</div>
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
        max_p   = float(df_s["probability"].max())

        k1, k2, k3, k4 = st.columns(4)
        for col, val, label_kpi, color in [
            (k1, total,   "Total patients",  "#e2e8f0"),
            (k2, at_risk, "À risque",        "#e53e3e"),
            (k3, safe,    "Sains",           "#38a169"),
            (k4, f"{avg_p:.0%}", "Risque moyen", "#7c8db5"),
        ]:
            with col:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-value" style="color:{color};">{val}</div>'
                    f'<div class="kpi-label">{label_kpi}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Répartition des résultats")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#13162b")
            ax.set_facecolor("#13162b")
            if at_risk == 0:
                sizes, labels_pie = [safe], ["Sains"]
                colors_pie = ["#38a169"]
            elif safe == 0:
                sizes, labels_pie = [at_risk], ["À risque"]
                colors_pie = ["#e53e3e"]
            else:
                sizes, labels_pie = [at_risk, safe], ["À risque", "Sains"]
                colors_pie = ["#e53e3e", "#38a169"]
            wedges, _, autotexts = ax.pie(
                sizes, labels=labels_pie, colors=colors_pie,
                autopct="%1.0f%%", startangle=90,
                textprops={"color": "white", "fontsize": 11},
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax.set_title("Répartition", color="white", fontsize=13, pad=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown("#### Probabilité par patient")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#13162b")
            ax.set_facecolor("#1a1f3c")
            probas = df_s["probability"].values
            bar_colors = ["#e53e3e" if p >= 0.5 else "#38a169" for p in probas]
            ax.bar(range(len(probas)), probas, color=bar_colors, alpha=0.85, edgecolor="none")
            ax.axhline(0.5, color="#a0aec0", linestyle="--", lw=1.2, alpha=0.8)
            ax.text(len(probas) - 0.5, 0.52, "seuil 50 %",
                    color="#a0aec0", fontsize=8, ha="right")
            ax.set_ylim(0, 1)
            ax.set_xlabel("Patient #", color="#a0aec0", fontsize=9)
            ax.set_ylabel("Probabilité", color="#a0aec0", fontsize=9)
            ax.tick_params(colors="#a0aec0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3561")
            ax.set_title("Évolution des probabilités", color="white", fontsize=13, pad=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if len(df_s) >= 3:
            st.markdown("#### Corrélation Âge / Probabilité de risque")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            fig.patch.set_facecolor("#13162b")
            ax.set_facecolor("#1a1f3c")
            sc = ax.scatter(df_s["age"], df_s["probability"],
                            c=df_s["probability"], cmap="RdYlGn_r",
                            s=90, alpha=0.85, edgecolors="none", vmin=0, vmax=1)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Probabilité", color="#a0aec0", fontsize=9)
            cb.ax.yaxis.set_tick_params(color="#a0aec0")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#a0aec0")
            ax.set_xlabel("Âge", color="#a0aec0")
            ax.set_ylabel("Probabilité de risque", color="#a0aec0")
            ax.tick_params(colors="#a0aec0")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3561")
            ax.set_title("Âge vs Risque cardiaque", color="white", fontsize=13, pad=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔧 État des services")

    # ── Services status ────────────────────────────────────────────────────────
    api_ok2, _   = get_api_health()
    mlf_ok, _    = get_mlflow_health()
    info2        = get_model_from_info = get_model_info() if api_ok2 else {}

    s1, s2, s3 = st.columns(3)

    def svc_card(col, icon, name, ok, detail, url):
        with col:
            color  = "#38a169" if ok else "#e53e3e"
            status = "● En ligne" if ok else "● Hors ligne"
            st.markdown(f"""
            <div class="kpi-card" style="text-align:left; padding:1.2rem 1.4rem;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-weight:700; font-size:1rem; margin:0.4rem 0 0.2rem;">
                    {name}
                </div>
                <div style="color:{color}; font-size:0.82rem; font-weight:600;">
                    {status}
                </div>
                <div style="color:#6b7fa8; font-size:0.78rem; margin-top:0.3rem;">
                    {detail}
                </div>
                <div style="margin-top:0.6rem;">
                    <a href="{url}" target="_blank"
                       style="color:#7c8db5; font-size:0.78rem;">
                        🔗 {url}
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

    svc_card(s1, "🤖", "API FastAPI",
             api_ok2,
             f"modèle : {info2.get('model_name','-')}",
             f"{API_PUBLIC_URL}/docs")
    svc_card(s2, "📊", "MLflow Tracking",
             mlf_ok,
             "expériences & model registry",
             MLFLOW_PUBLIC_URL)
    svc_card(s3, "🫀", "Frontend Streamlit",
             True,
             "cette interface",
             API_PUBLIC_URL.replace(":8000", ":8502").replace("/docs", ""))

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
        m_color = "#38a169" if method == "GET" else "#3182ce"
        st.markdown(f"""
        <div style="background:#1a1f3c; border:1px solid #2d3561; border-radius:10px;
                    padding:0.7rem 1.2rem; margin:0.4rem 0; display:flex;
                    align-items:center; gap:1rem;">
            <span style="background:{m_color}22; color:{m_color}; font-weight:700;
                         font-size:0.78rem; padding:3px 10px; border-radius:6px;
                         min-width:48px; text-align:center;">{method}</span>
            <span style="color:#e2e8f0; font-family:monospace; font-size:0.9rem;">
                {API_PUBLIC_URL}{path}
            </span>
            <span style="color:#6b7fa8; font-size:0.85rem; margin-left:auto;">
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



st.markdown(
    f"<hr style='border-color:#2d3561; margin-top:2rem;'>"
    f"<p style='text-align:center; color:#3d4f6e; font-size:0.78rem;'>"
    f"API : {API_PUBLIC_URL} &nbsp;|&nbsp; MLflow : {MLFLOW_PUBLIC_URL}</p>",
    unsafe_allow_html=True,
)
