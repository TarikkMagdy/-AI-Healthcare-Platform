import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
index_label = {
    "0": "Acne and Rosacea Photos",
    "1": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "2": "Atopic Dermatitis Photos",
    "3": "Cellulitis Impetigo and other Bacterial Infections",
    "4": "Light Diseases and Disorders of Pigmentation",
    "5": "Lupus and other Connective Tissue diseases",
    "6": "Poison Ivy Photos and other Contact Dermatitis",
    "7": "Psoriasis pictures Lichen Planus and related diseases",
    "8": "Seborrheic Keratoses and other Benign Tumors",
    "9": "Systemic Disease",
    "10": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "11": "Vascular Tumors",
    "12": "Warts Molluscum and other Viral Infections"
}
OUT_CLASSES = len(index_label)
label_from_index = {int(k): v for k, v in index_label.items()}

class_descriptions = {
    "Acne and Rosacea Photos": (
        "These conditions primarily affect the face and include acne (characterized by blackheads, whiteheads, pimples, cysts, and possible scarring) "
        "and rosacea (chronic redness, visible blood vessels, and inflammatory bumps). Triggers may include hormonal changes, stress, diet, or sun exposure. "
        "Acne often begins during adolescence, while rosacea typically affects adults."
    ),
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": (
        "Includes sun-induced pre-cancerous lesions (Actinic Keratosis), the most common type of skin cancer (Basal Cell Carcinoma), and other malignant "
        "lesions like Squamous Cell Carcinoma and Melanoma. These conditions may appear as scaly patches, open sores, or abnormal growths and often require "
        "biopsy and medical intervention."
    ),
    "Atopic Dermatitis Photos": (
        "A chronic inflammatory skin disease, also known as eczema, causing dry, itchy, red, and cracked skin. Common in children but can affect all ages. "
        "Often linked to allergies, asthma, and hay fever, and may flare up with environmental triggers or stress."
    ),
    "Cellulitis Impetigo and other Bacterial Infections": (
        "These are bacterial infections of the skin. Cellulitis involves deeper layers, causing redness, swelling, warmth, and pain; it can become serious if untreated. "
        "Impetigo is a highly contagious, superficial infection characterized by honey-colored crusted sores, often around the nose and mouth. Caused by Staphylococcus or Streptococcus bacteria."
    ),
    "Light Diseases and Disorders of Pigmentation": (
        "Conditions affected by light (e.g., Polymorphic Light Eruption) or involving abnormal pigmentation. This includes vitiligo (loss of pigment), melasma (brown patches), and post-inflammatory "
        "hyperpigmentation. May be genetic, autoimmune, or related to sun exposure."
    ),
    "Lupus and other Connective Tissue diseases": (
        "Autoimmune disorders where the immune system attacks the body’s connective tissues, such as skin, joints, and internal organs. Lupus can cause the signature 'butterfly rash' on the face, "
        "as well as systemic symptoms like fatigue and joint pain. Other conditions include dermatomyositis and scleroderma."
    ),
    "Poison Ivy Photos and other Contact Dermatitis": (
        "Irritant or allergic skin reactions resulting from exposure to allergens like poison ivy, nickel, or harsh chemicals. Symptoms include red, itchy rash, blisters, or swelling. "
        "Avoidance and topical treatments (like corticosteroids) are common management strategies."
    ),
    "Psoriasis pictures Lichen Planus and related diseases": (
        "Chronic inflammatory skin diseases with autoimmune origins. Psoriasis presents with thick, scaly plaques, commonly on elbows, knees, and scalp. "
        "Lichen Planus appears as purplish, itchy, flat-topped bumps that may affect skin, nails, and mucous membranes. Both can be triggered by stress, medications, or infections."
    ),
    "Seborrheic Keratoses and other Benign Tumors": (
        "Non-cancerous skin growths that often appear with aging. Seborrheic Keratoses are wart-like, waxy, or scaly in texture and can range in color. "
        "Other benign lesions include lipomas, dermatofibromas, and skin tags. These typically require no treatment unless symptomatic or for cosmetic reasons."
    ),
    "Systemic Disease": (
        "Refers to skin signs of internal systemic diseases, such as jaundice from liver disease, acanthosis nigricans from diabetes, or skin thickening in scleroderma. "
        "Early skin findings can provide critical clues to diagnosing potentially serious underlying conditions."
    ),
    "Tinea Ringworm Candidiasis and other Fungal Infections": (
        "Caused by dermatophytes (ringworm/tinea), yeasts (candidiasis), or molds. Symptoms may include itchy, scaly patches, redness, or ring-like lesions. "
        "Candidiasis affects moist areas like the mouth, genitals, and skin folds. Contagious and often treated with antifungal medications."
    ),
    "Vascular Tumors": (
        "Tumors originating from blood or lymphatic vessels. These include benign forms like infantile hemangiomas and more aggressive types like Kaposi’s sarcoma. "
        "They can appear as red or purple nodules or plaques and may require monitoring, laser treatment, or surgical removal depending on severity."
    ),
    "Warts Molluscum and other Viral Infections": (
        "Viral skin infections include warts (caused by human papillomavirus/HPV) and Molluscum contagiosum (caused by a poxvirus). "
        "Warts are rough, skin-colored lesions often found on hands and feet. Molluscum causes small, dome-shaped, umbilicated bumps. "
        "These are contagious and usually self-limiting but may be treated for cosmetic or comfort reasons."
    )
}



# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc.in_features = nn.Linear(num_ftrs, OUT_CLASSES)

    try:
        model.load_state_dict(torch.load("./models/skin_disease_classifier.pth"))
        print("✅ Loaded fine-tuned weights from skin_disease_classifier.pth.")
    except FileNotFoundError:
        print("⚠️ WARNING: skin_disease_classifier.pth not found.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")

    model = model.to(device)
    model.eval()
    return model, weights.transforms()


model, preprocess_transforms = load_model()

# --- Image Preprocessing ---
def transform_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return preprocess_transforms(img).unsqueeze(0)
    except Exception as e:
        print(f"Error transforming image: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process image file. Error: {e}")

# New: Preprocessing from image path for testing
def transform_image_from_path(image_path: str):
    try:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        print(f"Error processing image from path: {e}")
        raise

# --- Prediction ---
def get_prediction(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_idx = outputs.argmax(1).item()
    return predicted_idx

# --- FastAPI App ---
app = FastAPI(title="Skin Condition Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    predicted_class: str
    description: str

@app.post("/predict/", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image_tensor = transform_image(image_bytes)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error during image transform: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during image processing.")

    try:
        predicted_index = get_prediction(image_tensor)
        predicted_label = label_from_index.get(predicted_index, "Unknown Class")
        description = class_descriptions.get(predicted_label, "No description available.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    return PredictionResponse(
        predicted_class=predicted_label,
        description=description
    )

# --- Optional: Predict from local image path ---
@app.get("/predict-from-path/", response_model=PredictionResponse)
def predict_from_path(image_path: str):
    try:
        image_tensor = transform_image_from_path(image_path)
        predicted_index = get_prediction(image_tensor)
        predicted_label = label_from_index.get(predicted_index, "Unknown Class")
        description = class_descriptions.get(predicted_label, "No description available.")
    except Exception as e:
        print(f"Error during prediction from path: {e}")
        raise HTTPException(status_code=500, detail="Prediction from path failed.")

    return PredictionResponse(
        predicted_class=predicted_label,
        description=description
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
