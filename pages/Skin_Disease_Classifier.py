import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000/predict/"

st.set_page_config(page_title="Skin Condition Classifier", layout="centered")

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        margin-top: 1em;
    }
    .subtitle {
        text-align: center;
        color: #4f4f4f;
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 10px;
        font-weight: 500;
        padding: 0.5em 1.2em;
    }
    .stButton>button:hover {
        background-color: #218838;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🔬 Skin Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a skin image to detect the condition using an AI model.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    if st.button("🔍 Predict"):
        with st.spinner("Classifying..."):
            try:
                response = requests.post(API_URL, files={"file": image_bytes})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"🧠 Prediction: **{result['predicted_class']}**")
                    st.markdown(f"📖 **Description:**\n\n{result['description']}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
