from torchvision import models
import torch.nn as nn
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
from torchvision import models

model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2) 


# Page config
st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# Title & Style
st.markdown("""
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
    background-color: #007BFF;
    color: white;
    border-radius: 10px;
    font-weight: 500;
    padding: 0.5em 1.2em;
}
.stButton>button:hover {
    background-color: #0056b3;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🩺 Pneumonia Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a chest X-ray image to detect pneumonia using EfficientNet.</div>', unsafe_allow_html=True)

# ---- Load Model ----
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Binary classifier
    model.load_state_dict(torch.load(r"C:\Users\mh223\3D Objects\AI Clinic\models\pneumonia_efficientnet.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
class_names = ["Normal", "Pneumonia"]

# ---- Preprocessing ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet default size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_idx = torch.argmax(outputs, dim=1).item()
                predicted_class = class_names[predicted_idx]

            st.success(f"🧠 Prediction: **{predicted_class}**")
            if predicted_class == "Pneumonia":
                st.error("⚠️ Pneumonia detected. Consult a medical professional.")
            else:
                st.success("✅ No signs of pneumonia detected.")

