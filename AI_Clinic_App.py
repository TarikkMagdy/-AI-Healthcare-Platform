import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie 
import json 

# Function to load Lottie animation from local file
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Lottie file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding Lottie JSON from {filepath}")
        return None


# Page Configuration
st.set_page_config(
    page_title="AI Clinic | Digital Health Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    logo = Image.open("assets/logo.png")
    st.sidebar.image(logo, width=200)
except FileNotFoundError:
    st.sidebar.warning("Logo image not found. Please add 'logo.png' in 'assets' folder.")


# Custom CSS Styling (keep your existing CSS)
st.markdown("""
<style>
    :root {
        --primary: #2a7f62;
        --secondary: #3ab7ca;
        --accent: #ff6b6b;
        --light: #f8f9fa;
        --dark: #343a40;
    }


    .main-title {
        text-align: center;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        color: var(--primary);
        font-weight: 700;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-text {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 1.5rem; /* Reduced margin a bit */
        color: var(--dark);
        line-height: 1.6;
    }

    .service-card {
        border-radius: 12px;
        padding: 1.5rem;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%; /* Ensures cards in a row have the same height */
        display: flex; /* Added for flex layout */
        flex-direction: column; /* Added for flex layout */
        justify-content: space-between; /* Distributes space for button */
        border: none !important;
    }

    .service-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .service-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .service-desc {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.5rem;
        min-height: 60px; /* Keep this or adjust as needed */
        flex-grow: 1; /* Allows description to take available space */
    }
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(42, 127, 98, 0.3);
        color: white;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }

    .footer {
        font-size: 0.85rem;
        text-align: center;
        color: #6c757d;
        margin-top: 2rem;
    }

    /* Optional: Style for Lottie container */
    .lottie-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem; /* Add some space below the animation */
    }
</style>
""", unsafe_allow_html=True)

# Sidebar (keep your existing sidebar code)
st.sidebar.title("🩺 AI Clinic Services")
st.sidebar.markdown("---")
st.sidebar.info("""
**Welcome to your digital health companion!**
Our AI-powered tools provide preliminary insights.
Always consult a healthcare professional for medical advice.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Feedback")
feedback = st.sidebar.text_area("Your thoughts?")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thanks for your feedback!")

st.sidebar.markdown("""
<div class="footer">
    © 2024-2025 AI Clinic<br>
    All Rights Reserved
</div>
""", unsafe_allow_html=True)


# Title and Subtitle
st.markdown('<div class="main-title">AI Clinic Digital Health Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Empowering your health journey with intelligent diagnostics and support tools.<br>Select a service below to begin your analysis.</div>', unsafe_allow_html=True)

# --- Lottie Animation ---

lottie_animation_path = "assets/health_animation.json" 
lottie_json = load_lottiefile(lottie_animation_path)

if lottie_json:
    with st.container(): # Use a container for better layout control
        st.markdown('<div class="lottie-container">', unsafe_allow_html=True) 
        st_lottie(
            lottie_json,
            speed=1,
            reverse=False,
            loop=True,
            quality="high", # medium ; high
            height=250,  # Adjust height
            width=250,   # Adjust width
            key="health_animation"
        )
        st.markdown('</div>', unsafe_allow_html=True) # Close the lottie-container div

# Services
st.subheader("✨ Our AI Health Services")
cols = st.columns(5) 


with cols[0]:
    st.markdown("""
    <div class="service-card">
        <div> <div class="service-title">🧠 AI Therapist</div>
            <div class="service-desc">
                Your compassionate digital companion for mental wellness and emotional support.
            </div>
        </div>
    """, unsafe_allow_html=True) # Button will be added below this div by Streamlit
    if st.button("Start Therapy Session", key="therapist_go", help="Speak with your AI Therapist"):
        with st.spinner("Loading AI Therapist..."):
            st.switch_page("pages/AI_Therapist.py")
    st.markdown("</div>", unsafe_allow_html=True) # Close .service-card

with cols[1]:
    st.markdown("""
    <div class="service-card">
        <div>
            <div class="service-title">🧴 Skincare AI</div>
            <div class="service-desc">
                Personalized skincare analysis and product recommendations through image recognition.
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Analyze My Skin", key="skincare_go", help="Upload your photo for skincare advice"):
        with st.spinner("Analyzing skin..."):
            st.switch_page("pages/Skincare_AI.py")
    st.markdown("</div>", unsafe_allow_html=True)

with cols[2]:
    st.markdown("""
    <div class="service-card">
        <div>
            <div class="service-title">🔬 Skin Analyzer</div>
            <div class="service-desc">
                Identify common skin conditions through advanced image classification technology.
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Diagnose Skin Condition", key="skin_classifier_go", help="Check for skin disease"):
        with st.spinner("Diagnosing..."):
            st.switch_page("pages/Skin_Disease_Classifier.py")
    st.markdown("</div>", unsafe_allow_html=True)

with cols[3]:
    st.markdown("""
    <div class="service-card">
        <div>
            <div class="service-title">❤️ Heart Health</div>
            <div class="service-desc">
                Comprehensive cardiovascular risk assessment using clinical parameters and AI.
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Check Heart Health", key="heart_classifier_btn", help="Cardiovascular risk analysis"):
        with st.spinner("Assessing heart health..."):
            st.switch_page("pages/Heart_Disease_classifier.py")
    st.markdown("</div>", unsafe_allow_html=True)

with cols[4]:
    st.markdown("""
    <div class="service-card">
        <div>
            <div class="service-title">🫁 Lung Classifier</div>
            <div class="service-desc">
                Pneumonia detection from chest X-ray images using EfficientNet and deep learning.
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("Check Lung Health", key="lung_classifier_btn", help="Detect pneumonia via X-ray"):
        with st.spinner("Analyzing lung X-ray..."):
            st.switch_page("pages/Pneumonia_Detector.py")
    st.markdown("</div>", unsafe_allow_html=True)


# Disclaimer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
    ⚠️ All AI analyses are preliminary and should be verified by a medical professional.
</div>
""", unsafe_allow_html=True)