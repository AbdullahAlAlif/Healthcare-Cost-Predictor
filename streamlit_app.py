import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Set page config
st.set_page_config(
    page_title="Healthcare Cost Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load the model
@st.cache_resource
def load_model():
    return pickle.load(open('Healt_insurace_charge_model.sav', 'rb'))

model = load_model()

def minmax_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

st.title("💰 Healthcare Cost Predictor")
st.markdown("""
Welcome! Enter your details below to predict your annual healthcare cost.\
This app uses a machine learning model trained on real insurance data.
""")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=30, help="Select your age in years.")
    with col2:
        bmi = st.slider("BMI", min_value=15.0, max_value=40.0, value=25.0, step=0.1, help="Select your Body Mass Index.")
    smoker = st.toggle("Are you a smoker?", value=False)
    submitted = st.form_submit_button("Predict Cost", use_container_width=True)

def get_feature_impact(age, bmi, smoker, model):
    # Baseline: all features at minimum
    base = np.array([[0, 0, 0, 0, 0, 0]])
    base_pred = model.predict(base)[0]
    # Individual feature impact
    age_scaled = minmax_scale(age, 18, 100)
    bmi_scaled = minmax_scale(bmi, 15, 40)
    smoker_int = int(smoker)
    features = np.array([[age_scaled, bmi_scaled, smoker_int, age_scaled * bmi_scaled, age_scaled * smoker_int, bmi_scaled * smoker_int]])
    pred = model.predict(features)[0]
    # Calculate impact by toggling each feature
    impacts = {}
    feature_names = ["Age", "BMI", "Smoker", "Age*BMI", "Age*Smoker", "BMI*Smoker"]
    for i, name in enumerate(feature_names):
        test = base.copy()
        test[0, i] = features[0, i]
        test_pred = model.predict(test)[0]
        impacts[name] = test_pred - base_pred
    return impacts, pred, base_pred

if submitted:
    smoker_int = int(smoker)
    age_scaled = minmax_scale(age, 18, 100)
    bmi_scaled = minmax_scale(bmi, 15, 40)
    features = np.array([[age_scaled, bmi_scaled, smoker_int, age_scaled * bmi_scaled, age_scaled * smoker_int, bmi_scaled * smoker_int]])
    prediction = model.predict(features)[0]
    # Personalized Suggestion
    bmi_reduced = max(15, bmi-2)
    bmi_reduced_scaled = minmax_scale(bmi_reduced, 15, 40)
    features_bmi_reduced = np.array([[age_scaled, bmi_reduced_scaled, smoker_int, age_scaled * bmi_reduced_scaled, age_scaled * smoker_int, bmi_reduced_scaled * smoker_int]])
    pred_bmi_reduced = model.predict(features_bmi_reduced)[0]
    savings = prediction - pred_bmi_reduced
    # Breakdown & Summary in big text
    st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center; margin-top: 2em;'>
        <div style='background: #f8f9fa; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 2em 3em; text-align: center;'>
            <h2 style='color: #007bff; margin-bottom: 0.5em;'>Predicted Healthcare Cost</h2>
            <p style='font-size: 2.5em; font-weight: bold; color: #28a745;'>${prediction:,.2f} <span style="font-size:0.5em; color:#333;">/year</span></p>
            <p style='font-size: 1.5em; color: #555;'>Monthly: <b>${prediction/12:,.2f}</b></p>
            <p style='font-size: 1.2em; color: #222;'>At age <b>{age}</b> with a BMI of <b>{bmi}</b> and as a <b>{'smoker' if smoker else 'non-smoker'}</b>, your estimated annual cost is <b>${prediction:,.0f}</b>.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Add some vertical space before the suggestion
    st.markdown("<div style='margin-top: 2em;'></div>", unsafe_allow_html=True)
    # Personalized Suggestion & Motivation
    threshold_high = 12000  # You can adjust this threshold as needed

    # BMI advice
    if bmi > 25:
        st.info(
            f"**BMI Advice:** While reducing your BMI by 2 points could save you about ${savings:,.0f}/year, "
            "the impact on your healthcare cost is moderate. However, maintaining a healthy BMI can greatly improve your daily life, "
            "boost your energy, and reduce the risk of chronic diseases."
        )
    else:
        st.info(
            "Your BMI is in a healthy range. Keeping it steady supports your overall well-being."
        )

    # Smoking advice
    if smoker:
        st.warning(
            "**Smoking Impact:** Smoking significantly increases your healthcare costs and has a major negative effect on your health. "
            "Quitting smoking can greatly reduce your expenses and improve your quality of life."
        )
    else:
        st.success(
            "Staying smoke-free helps keep your healthcare costs low and protects your long-term health."
        )

    # General cost summary
    if prediction >= threshold_high:
        st.warning("Your predicted healthcare cost is on the higher side. Consider the above suggestions to help lower your expenses.")
    else:
        st.success("Great job! Your predicted healthcare cost is relatively low. Keep up the healthy habits!")
    
    # Downloadable Report
    def create_pdf():
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, "Healthcare Cost Prediction Report")
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, f"Age: {age}")
        c.drawString(72, 700, f"BMI: {bmi}")
        if smoker:
            c.drawString(72, 680, "Smoker: Yes (High risk; cost impact significant; Quitting strongly recommended)")
        else:
            c.drawString(72, 680, "Smoker: No")
        c.drawString(72, 660, f"Predicted Annual Cost: ${prediction:,.2f}")
        c.drawString(72, 640, f"Monthly Cost: ${prediction/12:,.2f}")
        
        c.save()
        buffer.seek(0)
        return buffer
    st.download_button("Download PDF Report", data=create_pdf(), file_name="healthcare_cost_report.pdf", mime="application/pdf")
else:
    st.info("Fill in the form and click 'Predict Cost' to see your result.")

st.markdown("---")
st.caption("Made with Streamlit · Model: Linear Regression · Data: insurance.csv")
