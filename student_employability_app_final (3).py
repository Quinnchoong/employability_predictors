
# -*- coding: utf-8 -*-
# student_employability_app_final.py - Final Clean Version

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import base64
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- Load Model & Scaler ---
with open("employability_predictor.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Utility Functions ---
def generate_pdf_report(data, result, confidence, proba):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Employability Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for k, v in data.items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Probabilities: Employable {proba[1]*100:.2f}%, Less Employable {proba[0]*100:.2f}%", ln=True)
    file_path = "prediction_report.pdf"
    pdf.output(file_path)
    return file_path

def get_pdf_download_link(file_path):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="prediction_report.pdf">📄 Download PDF Report</a>'
    return href

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="Graduate Employability Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/UNESCO_logo.svg/2560px-UNESCO_logo.svg.png",
    width=200
)

st.sidebar.title("About This App")
st.sidebar.markdown(
    '''
    This app predicts **graduate employability** based on academic and experiential attributes.
    Outputs: Prediction, confidence, feature insights, downloadable PDF report.
    ---
    Developed for MSc Capstone Project.
    '''
)
st.sidebar.info("Version: Final | Last Updated: 2025-07-20")

st.title("🎓 Advanced Graduate Employability Dashboard")
st.subheader("Empowering HEIs with actionable, data-driven insights.")

tab1, tab2, tab3 = st.tabs(["📋 Input Form", "📊 Feature Insights", "📄 Report"])

# ---------------- Tab 1: Input Form ----------------
with tab1:
    st.header("📋 Student Profile Input")

    with st.form("input_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender (0=Female, 1=Male)", [0,1])
            general_appearance = st.slider("General Appearance (1–5)", 1, 5, 3)
            gpa = st.number_input("General Point Average (GPA)", 0.0, 4.0, 3.0)

        with col2:
            manner_of_speaking = st.slider("Manner of Speaking (1–5)", 1, 5, 3)
            physical_condition = st.slider("Physical Condition (1–5)", 1, 5, 3)
            mental_alertness = st.slider("Mental Alertness (1–5)", 1, 5, 3)

        with col3:
            self_confidence = st.slider("Self-Confidence (1–5)", 1, 5, 3)
            ability_to_present_ideas = st.slider("Ability to Present Ideas (1–5)", 1, 5, 3)
            communication_skills = st.slider("Communication Skills (1–5)", 1, 5, 3)
            student_performance_rating = st.slider("Student Performance Rating (1–5)", 1, 5, 3)
            no_skills = st.slider("Number of Skills (integer)", 0, 10, 0)
            year_of_graduate = st.selectbox("Year of Graduation", [2020,2021,2022,2023,2024])

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        feature_names = [
            'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
            'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
            'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
            'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
        ]

        input_data = np.array([[gender, general_appearance, gpa,
                                manner_of_speaking, physical_condition, mental_alertness,
                                self_confidence, ability_to_present_ideas, communication_skills,
                                student_performance_rating, no_skills, year_of_graduate]])

        if input_data.shape[1] != len(feature_names):
            st.error(f"Input shape mismatch: expected {len(feature_names)}, got {input_data.shape[1]}")
        else:
            input_scaled = scaler.transform(input_data)
            proba = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]

            st.write("🎓 The system predicts if the student is employable or less employable.")

            if prediction == 1:
                result_text = "✅ The student is predicted to be **Employable**"
                result_color = "green"
            else:
                result_text = "⚠️ The student is predicted to be **Less Employable**"
                result_color = "red"

            confidence = proba[prediction] * 100

            st.session_state['data'] = dict(zip(feature_names, input_data[0]))
            st.session_state['result'] = result_text
            st.session_state['confidence'] = confidence
            st.session_state['proba'] = proba

            st.markdown("---")
            st.markdown(f"<h3 style='color:{result_color}'>{result_text}</h3>", unsafe_allow_html=True)

            st.write(f"📊 **Probabilities:**")
            st.write(f"Probability of being Employable: {proba[1]*100:.2f}%")
            st.write(f"Probability of being Less Employable: {proba[0]*100:.2f}%")

# ---------------- Tab 2: Feature Insights ----------------
with tab2:
    st.header("📊 Feature Contribution")

    if 'data' in st.session_state:
        df = pd.DataFrame([st.session_state['data']])
        df.T.plot(kind="barh", legend=False, figsize=(8, 6), color='skyblue')
        plt.xlabel("Feature Value")
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.info("Please submit a prediction first on the 📋 Input Form tab.")

# ---------------- Tab 3: Report ----------------
with tab3:
    st.header("📄 Downloadable Prediction Report")

    if 'result' in st.session_state:
        pdf_path = generate_pdf_report(
            st.session_state['data'],
            st.session_state['result'],
            st.session_state['confidence'],
            st.session_state['proba']
        )
        st.markdown(get_pdf_download_link(pdf_path), unsafe_allow_html=True)
    else:
        st.info("Please submit a prediction first on the 📋 Input Form tab.")

st.markdown("---")
st.caption("© 2025 Your Name / Your University | Graduate Employability Prediction App | For research purposes only.")
