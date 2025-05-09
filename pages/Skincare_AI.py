# pages/02_Skincare_AI.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models import resnet101, ResNet101_Weights
import torch.nn as nn
import time


st.markdown("""
    <style>
        /* Applying to elements within this page if possible, or ensure selectors are specific */
        /* For instance, wrap content in a div with a class and target that */
        .skincare-page-container h3 { color: #31C48D; } /* Example for page-specific h3 */
    </style>
""", unsafe_allow_html=True)


# Load the dataset
@st.cache_data
def load_skincare_data():
    try:
        df = pd.read_csv("data/face_products2.csv")
        df["price"] = df["price"].astype(str).str.replace("£", "", regex=False).astype(float)
        return df.copy() 
    except FileNotFoundError:
        st.error("Skincare product data file (data/face_products2.csv) not found. Please check the path.")
        return pd.DataFrame() # Return empty DataFrame

df_products_full = load_skincare_data()

st.sidebar.title("🧴 Skincare Analysis Filters") 
if not df_products_full.empty:
    manual_skin_type_options = ["Auto Detect", "Oily", "Dry", "Normal"] + list(df_products_full['Skin Type'].unique())
    manual_skin_type_options = sorted(list(set(option for option in manual_skin_type_options if pd.notna(option)))) # Unique and sorted
    if "Auto Detect" not in manual_skin_type_options: manual_skin_type_options.insert(0,"Auto Detect")

    manual_skin_type = st.sidebar.selectbox("Select Skin Type (Optional)", manual_skin_type_options)
    price_filter = st.sidebar.slider("Filter by Price Range ($)",
                                     float(df_products_full["price"].min()),
                                     float(df_products_full["price"].max()),
                                     (float(df_products_full["price"].min()), float(df_products_full["price"].max())) # Default to full range
                                    )
else:
    st.sidebar.warning("Product data not loaded, filters unavailable.")
    manual_skin_type = "Auto Detect"
    price_filter = (0.0, 100.0)


# Load ML models
@st.cache_resource
def load_skin_models():
    try:
        skin_type_model_loaded = torch.load('models/skin_type_model_complete.pth', map_location=torch.device('cpu'))
        skin_type_model_loaded.eval()

        concern_model_loaded = resnet101(weights=None) 
        num_ftrs = concern_model_loaded.fc.in_features
        concern_model_loaded.fc = nn.Linear(num_ftrs, 4) 
        concern_model_loaded.load_state_dict(torch.load("models/best_model_concern___.pth", map_location=torch.device('cpu')))
        concern_model_loaded.eval()
        return skin_type_model_loaded, concern_model_loaded
    except FileNotFoundError as e:
        st.error(f"Skincare AI model file not found. Please check paths. Missing: {e.filename}")
        return None, None
    except Exception as e:
        st.error(f"Error loading skincare models: {e}")
        return None, None


skin_type_model, concern_model = load_skin_models()

if not df_products_full.empty and skin_type_model and concern_model:
    skincare_products_df = df_products_full.dropna(subset=['Concern List_']).copy()
    skincare_products_df['Skin Type'] = skincare_products_df['Skin Type'].fillna('Normal')

    # One-hot encoding for skin type
    encoder_skin = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    skin_type_encoded = encoder_skin.fit_transform(skincare_products_df[['Skin Type']])
    skin_type_df = pd.DataFrame(skin_type_encoded, columns=encoder_skin.get_feature_names_out(['Skin Type']))

    # TF-IDF for ingredient similarity
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    ingredients_tfidf = tfidf_vectorizer.fit_transform(skincare_products_df['ingredients'].fillna(''))
else:
    skincare_products_df = pd.DataFrame()


# Image Preprocessing
def preprocess_image(image_pil):
    preprocess_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess_transform(image_pil).unsqueeze(0)

# Function for product recommendation
def recommend_products(user_skin_type, user_concern, current_ingredients_tfidf, current_skincare_products_df, current_skin_type_df, current_encoder_skin, current_tfidf_vectorizer):
    if current_skincare_products_df.empty:
        return pd.DataFrame()

    user_skin_type_encoded = current_encoder_skin.transform([[user_skin_type]])

    all_concerns_list = sorted(list(current_skincare_products_df['Concern List_'].str.split(',\s*').explode().str.strip().unique()))
    if not all_concerns_list: 
        all_concerns_list = ['Acne', 'Bags', 'Enlarged pores', 'Redness']

    user_concern_vector_ingredients = current_tfidf_vectorizer.transform([user_concern]) 

    # Similarity for skin type (one-hot)
    skin_type_similarity = cosine_similarity(user_skin_type_encoded, current_skin_type_df)

    # Similarity for concern based on ingredients (TF-IDF)
    concern_similarity_ingredients = cosine_similarity(user_concern_vector_ingredients, current_ingredients_tfidf)
    
    # Combine similarities (you can adjust weights)
    final_similarity = (skin_type_similarity.flatten() * 0.5) + (concern_similarity_ingredients.flatten() * 0.5)

    # Get top recommendations
    # Ensure we don't request more items than available
    num_recommendations = min(5, len(current_skincare_products_df))
    if num_recommendations == 0:
        return pd.DataFrame()
        
    recommended_indices = final_similarity.argsort()[-num_recommendations:][::-1]
    recommended_df = current_skincare_products_df.iloc[recommended_indices]

    return recommended_df[
        (recommended_df["price"] >= price_filter[0]) &
        (recommended_df["price"] <= price_filter[1])
    ][['product_name', 'product_url', 'product_type', 'price', 'image_url']]


# --- Main UI Section for Skincare AI ---
st.title("Skincare Recommendations 🧴") # This title will appear at the top of the page
st.write("Upload an image to analyze your skin type and get personalized product recommendations.")

uploaded_file = st.file_uploader("Upload Your Skin Image", type=["jpg", "jpeg", "png"], key="skincare_uploader")

if uploaded_file and skin_type_model and concern_model and not skincare_products_df.empty:
    image = Image.open(uploaded_file).convert("RGB") # Ensure image is RGB
    st.image(image, caption="Uploaded Image", width=224) # Standardized width

    # Processing Animation
    with st.spinner('Analyzing your skin...'):
        # time.sleep(3) # Simulating processing time - can be removed for actual processing
        image_input = preprocess_image(image)

        # Predict skin type
        skin_type_labels_model = ["Oily", "Dry", "Normal"] # Labels model was trained on
        with torch.no_grad():
            skin_type_prediction_idx = torch.argmax(skin_type_model(image_input), dim=1).item()
        predicted_skin_type_from_model = skin_type_labels_model[skin_type_prediction_idx]
        
        final_skin_type = manual_skin_type if manual_skin_type != "Auto Detect" else predicted_skin_type_from_model
        st.success(f"✅ Skin Type for Recommendation: {final_skin_type} (Model predicted: {predicted_skin_type_from_model})")

        # Predict concern
        concern_labels_model = ['Acne', 'Bags', 'Enlarged pores', 'Redness'] # Labels model was trained on
        with torch.no_grad():
            concern_prediction_idx = torch.argmax(concern_model(image_input), dim=1).item()
        predicted_concern = concern_labels_model[concern_prediction_idx]
        st.warning(f"⚠️ Detected Concern for Recommendation: {predicted_concern}")

    # Recommendations
    st.write("### 🔥 Recommended Products for You")
    
    # Pass all necessary data to the recommendation function
    recommended_products_list = recommend_products(
        final_skin_type,
        predicted_concern, 
        ingredients_tfidf,
        skincare_products_df,
        skin_type_df,
        encoder_skin,
        tfidf_vectorizer
    )

    if not recommended_products_list.empty:
        # Display in columns, ensuring it handles less than 3 products
        num_products = len(recommended_products_list)
        cols = st.columns(min(num_products, 3) if num_products > 0 else 1)

        for idx, (_, row) in enumerate(recommended_products_list.iterrows()):
            col_idx = idx % (min(num_products, 3) if num_products > 0 else 1)
            with cols[col_idx]:
                st.markdown(f"""
                    <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px; height: 450px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <img src="{row['image_url'] if pd.notna(row['image_url']) else 'https://via.placeholder.com/150?text=No+Image'}"
                                 width="150" style="border-radius: 10px; max-height: 150px; object-fit: contain;" />
                            <h3 style="color: #31C48D; font-size: 1.1em; margin-top: 10px; height: 60px; overflow: hidden;">{row['product_name']}</h3>
                            <p style="font-size: 0.9em;"><strong>Type:</strong> {row['product_type']}</p>
                            <p style="color: #31C48D; font-size: 1.2em; font-weight: bold;">💲{row['price']:.2f}</p>
                        </div>
                        <a href="{row['product_url']}" target="_blank" style="text-decoration: none;">
                            <button style="background-color: #31C48D; color: white; padding: 10px 15px; border-radius: 5px; border: none; cursor: pointer; width: 100%; margin-top: 10px;">
                                View Product
                            </button>
                        </a>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No products found matching your criteria after filtering. Try adjusting the price range or other filters.")

elif uploaded_file and (not skin_type_model or not concern_model or skincare_products_df.empty):
    st.error("Models or product data could not be loaded. Recommendations are unavailable.")
elif not uploaded_file:
    st.info("Please upload a skin image to get started.")