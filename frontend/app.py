"""Frontend Streamlit — à compléter (TP S14bis).

Usage :
    streamlit run frontend/app.py --server.port 8501
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
        .hero {
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            background: linear-gradient(120deg, #f8d7da 0%, #fef3c7 100%);
            border: 1px solid rgba(0, 0, 0, 0.08);
            margin-bottom: 1rem;
        }
        .muted {
            color: #555;
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_health() -> tuple[bool, str]:
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        response.raise_for_status()
        return True, "API en ligne"
    except Exception:
        return False, "API indisponible"


left, right = st.columns([4, 1])
with left:
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">🫀 Heart Disease Classifier</h2>
            <p class="muted" style="margin:0.45rem 0 0 0;">
                Estime le risque de maladie cardiaque à partir de données cliniques.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    api_ok, api_msg = get_api_health()
    if api_ok:
        st.success(api_msg)
    else:
        st.error(api_msg)

with st.form("prediction_form"):
    st.subheader("Informations patient")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge (années)", min_value=1, max_value=120, value=52)
        trestbps = st.number_input("Pression artérielle au repos (mm Hg)", value=120)
        chol = st.number_input("Cholestérol sérique (mg/dl)", value=240)
        thalach = st.number_input("Fréquence cardiaque max atteinte", value=150)
        oldpeak = st.number_input("Dépression ST", value=1.0, step=0.1)
        ca = st.selectbox("Nb vaisseaux colorés (fluoroscopie)", [0, 1, 2, 3])
    with col2:
        sex = st.selectbox("Sexe", [1, 0], format_func=lambda x: "Homme" if x else "Femme")
        cp = st.selectbox("Type douleur thoracique", [1, 2, 3, 4])
        fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Résultats ECG au repos", [0, 1, 2])
        exang = st.selectbox("Angine induite par effort", [0, 1])
        slope = st.selectbox("Pente segment ST", [1, 2, 3])
        thal = st.selectbox("Thalassémie", [3, 6, 7])

    submitted = st.form_submit_button("Lancer la prédiction", use_container_width=True)

st.caption(f"API utilisée : {API_URL}")

if submitted:
    payload = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "ca": ca,
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "restecg": restecg,
        "exang": exang,
        "slope": slope,
        "thal": thal,
    }

    try:
        response = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        response.raise_for_status()
        result = response.json()

        label = int(result["label"])
        proba = float(result["probability"])

        st.subheader("Résultat")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.metric("Probabilité", f"{proba:.1%}")
        with c2:
            st.progress(min(max(proba, 0.0), 1.0))

        if label == 1:
            st.error(f"⚠️ Maladie cardiaque probable (probabilité : {proba:.1%})")
            st.info("Conseil : orienter vers une évaluation clinique approfondie.")
        else:
            st.success(f"✅ Pas de maladie cardiaque détectée (probabilité : {proba:.1%})")
            st.info("Conseil : conserver un suivi médical de routine.")
    except Exception as exc:
        st.error(f"Erreur d'appel API: {exc}")
