import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# 📋 CONFIG
st.set_page_config(page_title="🎓 Student Employability Predictor — Quick Fix", layout="centered")

# 📋 Load Model & Scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("employability_predictor.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# 📋 Header Image (optional)
try:
    image = Image.open("group-business-people-silhouette-businesspeople-abstract-background_656098-461.avif")
    st.image(image, use_container_width=True)
except:
    st.write("Header image not found — continuing without it.")

# 📋 Title & description
st.title("🎓 Student Employability Predictor — Quick Fix")
st.markdown("Fill in the input features and predict employability.")

# 🔷 Define feature columns (exact order & names from X_train_res)
feature_columns = [
    'GENDER', 'GENERAL_APPEARANCE', 'GENERAL_POINT_AVERAGE',
    'MANNER_OF_SPEAKING', 'PHYSICAL_CONDITION', 'MENTAL_ALERTNESS',
    'SELF-CONFIDENCE', 'ABILITY_TO_PRESENT_IDEAS', 'COMMUNICATION_SKILLS',
    'STUDENT_PERFORMANCE_RATING', 'NO_SKILLS', 'Year_of_Graduate'
]

# 🔷 Inputs
inputs = {}

col1, col2, col3 = st.columns(3)

with col1:
    inputs['GENDER'] = float(st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female", index=1))
    inputs['GENERAL_APPEARANCE'] = float(st.slider("Appearance (1-5)", 1, 5, 3))
    inputs['GENERAL_POINT_AVERAGE'] = float(st.number_input("GPA (0.0-4.0)", 0.0, 4.0, 3.0, 0.01))
    inputs['MANNER_OF_SPEAKING'] = float(st.slider("Speaking (1-5)", 1, 5, 3))

with col2:
    inputs['PHYSICAL_CONDITION'] = float(st.slider("Physical (1-5)", 1, 5, 3))
    inputs['MENTAL_ALERTNESS'] = float(st.slider("Alertness (1-5)", 1, 5, 3))
    inputs['SELF-CONFIDENCE'] = float(st.slider("Confidence (1-5)", 1, 5, 3))
    inputs['ABILITY_TO_PRESENT_IDEAS'] = float(st.slider("Ideas (1-5)", 1, 5, 3))

with col3:
    inputs['COMMUNICATION_SKILLS'] = float(st.slider("Communication (1-5)", 1, 5, 3))
    inputs['STUDENT_PERFORMANCE_RATING'] = float(st.slider("Performance (1-5)", 1, 5, 3))
    inputs['NO_SKILLS'] = float(st.radio("Has No Skills", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", index=0))
    inputs['Year_of_Graduate'] = float(st.number_input("Graduation Year", 2019, 2022, 2022))

# 🔷 Build input DataFrame in exact order
input_df = pd.DataFrame([inputs])[feature_columns]

st.subheader("📄 Input DataFrame (before scaling)")
st.write(input_df)

# 📋 Prediction
if st.button("Predict"):
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]

    st.subheader("📄 Scaled Input")
    st.write(pd.DataFrame(scaled_input, columns=input_df.columns))

    if pred == 1:
        st.success("🎉 The student is predicted to be **Employable**!")
        st.balloons()
    else:
        st.warning("⚠️ The student is predicted to be **Less Employable**.")

    st.info(f"Probability of being Employable: {proba[1]*100:.2f}%")
    st.info(f"Probability of being Less Employable: {proba[0]*100:.2f}%")

st.markdown("---")
st.caption(" Disclaimer: This prediction model is for research and informational purposes only.  
© 2025 Your Name / Your University | Graduate Employability Prediction App | For research purposes only.")
