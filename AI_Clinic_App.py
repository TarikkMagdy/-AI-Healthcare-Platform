import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AI Clinic | Welcome",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    logo = Image.open("assets/logo.png")
    st.sidebar.image(logo, width=200)
except FileNotFoundError:
    st.sidebar.warning("Logo image not found. Please add 'logo.png' in 'assets' folder.")

st.sidebar.title("AI Clinic Services")
st.sidebar.markdown("---")
st.sidebar.info(
    "Select a service from the navigation above to get started. "
    "Our AI models are for informational purposes and not a substitute for professional advice."
)
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024-2025 AI Clinic. All Rights Reserved.")

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.2em;
        margin-bottom: 0.2em;
    }
    .sub-text {
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 2em;
        color: #4f4f4f;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        font-weight: 500;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">Welcome to the AI Clinic 🩺</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Your intelligent partner in health and wellness. '
    'Our AI-powered tools are designed to provide insights and support.</div>',
    unsafe_allow_html=True
)

st.subheader("Explore Our Services:")

cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        st.markdown("##### 🧠 AI Therapist")
        st.markdown("A compassionate AI to listen and support you.")
        if st.button("Go to AI Therapist", key="therapist_go"):
            st.switch_page("pages/AI_Therapist.py")

with cols[1]:
    with st.container(border=True):
        st.markdown("##### 🧴 Skincare AI")
        st.markdown("Personalized skincare recommendations based on image analysis.")
        if st.button("Go to Skincare AI", key="skincare_go"):
            st.switch_page("pages/Skincare_AI.py")

with cols[2]:
    with st.container(border=True):
        st.markdown("##### 🔬 Skin Disease Classifier")
        st.markdown("Analyze skin images for insights on common conditions.")
        if st.button("Go to Skin Classifier", key="skin_classifier_go"):
            st.switch_page("pages/Skin_Disease_Classifier.py")

st.markdown("---")
