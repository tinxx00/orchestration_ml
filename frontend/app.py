"""Frontend Streamlit — à compléter (TP S14bis).

Usage :
    streamlit run frontend/app.py --server.port 8501
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="🫀",
    layout="centered",
)

st.title("🫀 Heart Disease Classifier")
st.markdown(
    "Prédit la probabilité de **maladie cardiaque** à partir de données cliniques."
)

# ---------------------------------------------------------------------------
# TODO (S14bis-1) : créer le formulaire avec les champs correspondant à
#                   NUMERIC_FEATURES + CATEGORICAL_FEATURES (voir config.py)
#
# with st.form("prediction_form"):
#     age      = st.number_input("Âge (années)", min_value=1, max_value=120, value=52)
#     trestbps = st.number_input("Pression artérielle au repos (mm Hg)", value=120)
#     chol     = st.number_input("Cholestérol sérique (mg/dl)", value=240)
#     thalach  = st.number_input("Fréquence cardiaque max atteinte", value=150)
#     oldpeak  = st.number_input("Dépression ST", value=1.0, step=0.1)
#     ca       = st.selectbox("Nb vaisseaux colorés (fluoroscopie)", [0, 1, 2, 3])
#     sex      = st.selectbox("Sexe", [1, 0], format_func=lambda x: "Homme" if x else "Femme")
#     cp       = st.selectbox("Type douleur thoracique", [1, 2, 3, 4])
#     fbs      = st.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1])
#     restecg  = st.selectbox("Résultats ECG au repos", [0, 1, 2])
#     exang    = st.selectbox("Angine induite par effort", [0, 1])
#     slope    = st.selectbox("Pente segment ST", [1, 2, 3])
#     thal     = st.selectbox("Thalassémie", [3, 6, 7])
#     submitted = st.form_submit_button("Prédire")
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (S14bis-2) : appeler l'API /predict avec httpx
#
# import httpx, os
# API_URL = os.getenv("API_URL", "http://localhost:8000")
#
# if submitted:
#     payload = {
#         "age": age, "trestbps": trestbps, "chol": chol,
#         "thalach": thalach, "oldpeak": oldpeak, "ca": ca,
#         "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg,
#         "exang": exang, "slope": slope, "thal": thal,
#     }
#     response = httpx.post(f"{API_URL}/predict", json=payload)
#     result = response.json()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (S14bis-3) : afficher le résultat de la prédiction
#
#     label = result["label"]
#     proba = result["probability"]
#     if label == 1:
#         st.error(f"⚠️ Maladie cardiaque probable (probabilité : {proba:.1%})")
#     else:
#         st.success(f"✅ Pas de maladie cardiaque détectée (probabilité : {proba:.1%})")
# ---------------------------------------------------------------------------

st.info("💡 Complétez les TODO de ce fichier lors du TP S14bis.")
