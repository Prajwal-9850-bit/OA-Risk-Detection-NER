import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OA Risk Check | NER",
    page_icon="🦴",
    layout="wide",
)

MODEL_PATH = Path("models/oa_risk_model.joblib")


def baseline_score(values: dict) -> tuple[int, list[str]]:
    """Transparent educational baseline; not a validated clinical score."""
    score = 0
    reasons = []
    rules = [
        (values["age"] >= 50, 2, "Age 50 years or above"),
        (values["bmi"] >= 30, 2, "BMI in the obesity range"),
        (values["previous_injury"], 2, "Previous knee, hip, or other joint injury"),
        (values["family_history"], 1, "Family history of osteoarthritis"),
        (values["joint_pain"], 2, "Current joint pain or stiffness"),
        (values["activity_limit"], 2, "Pain or stiffness limiting daily activity"),
        (values["heavy_work"], 1, "Repetitive heavy physical work"),
        (values["smoking"], 1, "Current tobacco use"),
    ]
    for present, points, reason in rules:
        if present:
            score += points
            reasons.append(reason)
    return min(score, 10), reasons


def risk_band(score: float) -> tuple[str, str]:
    if score < 3:
        return "Lower apparent risk", "success"
    if score < 6:
        return "Moderate apparent risk", "warning"
    return "Higher apparent risk", "error"


def model_prediction(values: dict):
    if not MODEL_PATH.exists():
        return None
    try:
        model = joblib.load(MODEL_PATH)
        row = pd.DataFrame([values])[model.feature_names_in_]
        probability = float(model.predict_proba(row)[0, 1])
        return probability
    except Exception:
        return None


st.title("🦴 OA Risk Check")
st.subheader("AI-assisted early detection prototype for the North Eastern Region")
st.warning(
    "This prototype provides a research/education risk estimate, not a diagnosis. "
    "Please consult a qualified clinician for persistent pain, swelling, injury, or disability."
)

with st.sidebar:
    st.header("About this MVP")
    st.write(
        "The form combines self-reported risk markers with a transparent baseline. "
        "If a locally trained model exists, the app will use it alongside the baseline."
    )
    st.caption("No form data is saved by this demo.")

with st.form("oa_risk_form"):
    st.markdown("### 1. Demographics and context")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=110, value=40, step=1)
        sex = st.selectbox("Sex (self-reported)", ["Female", "Male", "Other / prefer not to say"])
    with col2:
        height = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=165.0)
        weight = st.number_input("Weight (kg)", min_value=25.0, max_value=250.0, value=65.0)
    with col3:
        region = st.selectbox("NER state / region", [
            "Assam", "Arunachal Pradesh", "Manipur", "Meghalaya", "Mizoram",
            "Nagaland", "Sikkim", "Tripura", "Other / not specified",
        ])
        occupation = st.selectbox("Main occupation", ["Desk-based", "Agriculture", "Manual labour", "Household work", "Other"])

    bmi = weight / ((height / 100) ** 2)
    st.info(f"Calculated BMI: **{bmi:.1f}** (screening measure only; not a diagnosis)")

    st.markdown("### 2. Joint health markers")
    c1, c2 = st.columns(2)
    with c1:
        joint_pain = st.checkbox("Pain or stiffness in a knee, hip, hand, or other joint")
        activity_limit = st.checkbox("Joint symptoms limit walking, stairs, work, or daily activities")
        previous_injury = st.checkbox("Previous significant injury or surgery to a joint")
        family_history = st.checkbox("Close family member with osteoarthritis")
    with c2:
        heavy_work = st.checkbox("Frequent kneeling, squatting, lifting, or repetitive heavy work")
        smoking = st.checkbox("Current tobacco use")
        symptoms_months = st.slider("How long have symptoms lasted? (months)", 0, 60, 0)
        access = st.selectbox("Usual access to healthcare", ["Good", "Occasional difficulty", "Difficult / remote"])

    consent = st.checkbox("I understand this is not a diagnosis and want to view an educational estimate.")
    submitted = st.form_submit_button("Assess apparent risk", type="primary", use_container_width=True)

if submitted:
    if not consent:
        st.error("Please acknowledge the information notice before viewing the estimate.")
        st.stop()

    values = {
        "age": int(age), "bmi": bmi, "previous_injury": int(previous_injury),
        "family_history": int(family_history), "joint_pain": int(joint_pain),
        "activity_limit": int(activity_limit), "heavy_work": int(heavy_work),
        "smoking": int(smoking),
    }
    score, reasons = baseline_score(values)
    band, style = risk_band(score)
    probability = model_prediction(values)

    st.divider()
    st.markdown("### Result")
    result_col, detail_col = st.columns([1, 2])
    with result_col:
        getattr(st, style)(f"**{band}**")
        st.metric("Baseline marker score", f"{score}/10")
        if probability is not None:
            st.metric("Model-estimated probability", f"{probability:.0%}")
    with detail_col:
        if reasons:
            st.write("Markers contributing to this estimate:")
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("No selected markers increased the baseline estimate.")
        if score >= 3 or joint_pain:
            st.info("Consider arranging a non-urgent clinical assessment, especially if symptoms persist or worsen.")
        else:
            st.success("Maintain joint-friendly activity, healthy weight, and seek care if symptoms develop.")

    if symptoms_months >= 3 or activity_limit:
        st.warning("Persistent symptoms or functional limitation deserve clinical review. Seek urgent care after a serious injury, with a hot/red swollen joint, fever, or inability to bear weight.")

st.divider()
st.caption("Prototype for research planning. Risk estimates are not clinically validated and should not guide treatment or triage.")
