import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import joblib

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Page Header ---
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("""
This clinical decision support tool estimates the likelihood of cardiovascular disease based on patient health metrics.
The prediction is powered by a machine learning model trained on CDC behavioral risk factor data.
""")

# --- Mapping dictionaries ---
bmi_category_mapping = {
    'Underweight': 1, 'Normal weight': 0, 'Overweight': 2,
    'Obesity I': 3, 'Obesity II': 4, 'Obesity III': 5
}

age_category_mapping = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3,
    '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7,
    '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12
}

# --- Load Model ---
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    try:
        model = joblib.load(r"C:\Users\mh223\3D Objects\AI Clinic\models\Heart_Disease_Model.pkl")
        if hasattr(model, 'predict_proba'):
            return model
        raise ValueError("Model doesn't support probability predictions")
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# --- Input Form ---
with st.form("patient_assessment_form"):
    st.subheader("Patient Demographics")
    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        age_category = st.selectbox("Age Category", list(age_category_mapping.keys()), index=4)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)

    with demo_col2:
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 50.0, value=25.0)
        bmi_category = st.selectbox("BMI Classification", list(bmi_category_mapping.keys()), index=2)

    st.subheader("Lifestyle Factors")
    lifestyle_col1, lifestyle_col2 = st.columns(2)

    with lifestyle_col1:
        smoking = st.radio("Current Smoker", ["No", "Yes"], horizontal=True)
        alcohol = st.radio("Regular Alcohol Consumption", ["No", "Yes"], horizontal=True)
        physical_activity = st.radio("Physically Active", ["No", "Yes"], horizontal=True, index=1)

    with lifestyle_col2:
        difficulty_walking = st.radio("Difficulty Walking/Climbing", ["No", "Yes"], horizontal=True)
        sleep_time = st.slider("Average Nightly Sleep (hours)", 0, 24, 7)
        diabetic = st.selectbox("Diabetes Status", ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"])

    st.subheader("Recent Health Indicators")
    health_col1, health_col2 = st.columns(2)

    with health_col1:
        physical_health = st.slider("Poor Physical Health Days", 0, 30, 0)
        general_health = st.selectbox("Self-Reported General Health", ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'], index=2)

    with health_col2:
        mental_health = st.slider("Poor Mental Health Days", 0, 30, 0)

    st.subheader("Medical History")
    med_col1, med_col2, med_col3 = st.columns(3)

    with med_col1:
        stroke = st.radio("History of Stroke", ["No", "Yes"], horizontal=True)
        asthma = st.radio("Asthma Diagnosis", ["No", "Yes"], horizontal=True)

    with med_col2:
        kidney = st.radio("Chronic Kidney Disease", ["No", "Yes"], horizontal=True)

    with med_col3:
        skin_cancer = st.radio("Skin Cancer History", ["No", "Yes"], horizontal=True)

    submitted = st.form_submit_button("Assess Cardiovascular Risk")

# --- Prediction Logic ---
if submitted:
    model = load_model()

    if model:
        try:
            diabetic_status = 'Yes' if diabetic.startswith("Yes") else 'No'

            input_data = {
                'BMI_Category': bmi_category_mapping.get(bmi_category, -1),
                'AgeCategory': age_category_mapping.get(age_category, -1),
                'GenHealth': ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'].index(general_health),
                'Smoking_Yes': 1 if smoking == "Yes" else 0,
                'AlcoholDrinking_Yes': 1 if alcohol == "Yes" else 0,
                'Stroke_Yes': 1 if stroke == "Yes" else 0,
                'DiffWalking_Yes': 1 if difficulty_walking == "Yes" else 0,
                'Diabetic_Yes': 1 if diabetic_status == "Yes" else 0,
                'PhysicalActivity_Yes': 1 if physical_activity == "Yes" else 0,
                'Asthma_Yes': 1 if asthma == "Yes" else 0,
                'KidneyDisease_Yes': 1 if kidney == "Yes" else 0,
                'SkinCancer_Yes': 1 if skin_cancer == "Yes" else 0,
                'BMI': bmi,
                'PhysicalHealth': physical_health,
                'MentalHealth': mental_health,
                'Sex': 1 if sex == "Male" else 0,
                'SleepTime': sleep_time
            }

            input_df = pd.DataFrame([input_data])

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0][1]
                prediction = int(proba >= 0.5)
            else:
                prediction = int(model.predict(input_df)[0])
                proba = float(prediction)

            st.markdown("---")
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                risk_label = "High Risk ❗" if prediction == 1 else "Low Risk ✅"
                st.metric("Prediction", risk_label, delta=f"{proba:.0%} confidence")

            with col_res2:
                st.write("**Key Contributing Factors:**")
                top_factors = []
                if age_category in ['65-69', '70-74', '75-79', '80 or older']:
                    top_factors.append("Age")
                if bmi_category in ['Obesity II', 'Obesity III']:
                    top_factors.append("BMI")
                if smoking == "Yes":
                    top_factors.append("Smoking")
                if diabetic_status != "No":
                    top_factors.append("Diabetes")

                for factor in top_factors[:3]:
                    st.write(f"- {factor}")

            with st.expander("Detailed Probability Analysis"):
                st.progress(int(proba * 100))
                st.write(f"Probability of heart disease: {proba:.1%}")
                if proba > 0.7:
                    st.error("⚠️ High risk detected - consult a healthcare provider")
                elif proba > 0.4:
                    st.warning("Moderate risk - consider lifestyle changes")
                else:
                    st.success("Low risk - maintain healthy habits")

        except Exception as e:
            st.error("Prediction failed due to an unexpected error")
            st.error(f"Technical details: {str(e)}")
    else:
        st.error("Prediction unavailable - model failed to load")

# --- Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox, .stRadio, .stSlider {
        margin-bottom: 20px;
    }
    .risk-high {
        color: #d9534f;
        font-weight: bold;
    }
    .risk-moderate {
        color: #f0ad4e;
        font-weight: bold;
    }
    .risk-low {
        color: #5cb85c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)