import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import time

# -----------------------------
# Load Model & Classes
# -----------------------------
MODEL_PATH = "trained_plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

classes_list = [
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'
]

# Disease information dictionary (brief descriptions for educational purposes)
disease_info = {
    'Apple___Apple_scab': "Apple scab is a fungal disease causing olive-green spots on leaves and fruit. Treat with fungicides and prune affected areas.",
    'Apple___Black_rot': "Black rot affects apples with black spots on fruit and leaves. Remove infected parts and use fungicides.",
    'Apple___Cedar_apple_rust': "Cedar apple rust causes yellow-orange spots. Alternate hosts like cedar trees should be managed.",
    'Apple___healthy': "The apple plant is healthy with no signs of disease.",
    'Blueberry___healthy': "The blueberry plant is healthy with no signs of disease.",
    'Cherry_(including_sour)___Powdery_mildew': "Powdery mildew shows white powdery spots on leaves. Improve air circulation and apply fungicides.",
    'Cherry_(including_sour)___healthy': "The cherry plant is healthy with no signs of disease.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Gray leaf spot causes rectangular lesions on corn leaves. Use resistant varieties and fungicides.",
    'Corn_(maize)___Common_rust_': "Common rust produces reddish-brown pustules. Plant resistant hybrids.",
    'Corn_(maize)___Northern_Leaf_Blight': "Northern leaf blight shows cigar-shaped lesions. Rotate crops and use fungicides.",
    'Corn_(maize)___healthy': "The corn plant is healthy with no signs of disease.",
    'Grape___Black_rot': "Black rot causes mummified berries. Prune and apply fungicides early.",
    'Grape___Esca_(Black_Measles)': "Esca (Black Measles) leads to tiger-striped leaves. No cure; remove infected vines.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Leaf blight causes circular spots. Use fungicides and sanitation.",
    'Grape___healthy': "The grape plant is healthy with no signs of disease.",
    'Orange___Haunglongbing_(Citrus_greening)': "Citrus greening causes yellow shoots and malformed fruit. Manage psyllids and remove infected trees.",
    'Peach___Bacterial_spot': "Bacterial spot causes lesions on leaves and fruit. Use copper sprays and resistant varieties.",
    'Peach___healthy': "The peach plant is healthy with no signs of disease.",
    'Pepper,_bell___Bacterial_spot': "Bacterial spot on peppers shows dark spots. Avoid overhead watering and use copper.",
    'Pepper,_bell___healthy': "The bell pepper plant is healthy with no signs of disease.",
    'Potato___Early_blight': "Early blight causes concentric rings on leaves. Rotate crops and use fungicides.",
    'Potato___Late_blight': "Late blight leads to dark lesions and rot. Destroy infected plants and use protectants.",
    'Potato___healthy': "The potato plant is healthy with no signs of disease.",
    'Raspberry___healthy': "The raspberry plant is healthy with no signs of disease.",
    'Soybean___healthy': "The soybean plant is healthy with no signs of disease.",
    'Squash___Powdery_mildew': "Powdery mildew on squash shows white powder. Use resistant varieties and fungicides.",
    'Strawberry___Leaf_scorch': "Leaf scorch causes purple spots and scorching. Improve drainage and avoid stress.",
    'Strawberry___healthy': "The strawberry plant is healthy with no signs of disease.",
    'Tomato___Bacterial_spot': "Bacterial spot on tomatoes causes small spots. Use clean seeds and copper sprays.",
    'Tomato___Early_blight': "Early blight shows target-like spots. Stake plants and use mulch.",
    'Tomato___Late_blight': "Late blight causes large dark areas. Remove debris and use fungicides.",
    'Tomato___Leaf_Mold': "Leaf mold shows yellow patches. Increase ventilation and use resistant varieties.",
    'Tomato___Septoria_leaf_spot': "Septoria leaf spot causes small spots with gray centers. Prune and use fungicides.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Spider mites cause stippling and webbing. Use miticides and natural predators.",
    'Tomato___Target_Spot': "Target spot shows concentric circles. Remove lower leaves and use fungicides.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Yellow leaf curl virus stunts growth. Control whiteflies and use resistant varieties.",
    'Tomato___Tomato_mosaic_virus': "Mosaic virus causes mottled leaves. Use virus-free seeds and control aphids.",
    'Tomato___healthy': "The tomato plant is healthy with no signs of disease."
}

# -----------------------------
# Futuristic Styling with Custom CSS
# -----------------------------
st.markdown("""
    <style>
    /* Dark futuristic theme */
    .stApp {
        background-color: #0a0a0a;
        color: #00ff00;
    }
    /* Neon glow for titles */
    h1, h2, h3 {
        color: #00ffff;
        text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff;
    }
    /* Button styling */
    .stButton > button {
        background-color: #1a1a1a;
        color: #00ff00;
        border: 1px solid #00ff00;
        border-radius: 5px;
        box-shadow: 0 0 10px #00ff00;
    }
    .stButton > button:hover {
        background-color: #00ff00;
        color: #0a0a0a;
    }
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #00ff00;
        box-shadow: 0 0 10px #00ff00;
    }
    /* Expander */
    .streamlit-expanderHeader {
        color: #00ffff;
        text-shadow: 0 0 5px #00ffff;
    }
    /* Image caption */
    .css-1v3fvcr {
        color: #00ff00;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🌿🚀 Neo-Plant Disease Classifier")
st.write("Upload a leaf image or use your webcam for real-time analysis. Our AI will predict the disease with futuristic precision.")

# Sidebar for additional features
st.sidebar.title("🛠️ Control Panel")
use_webcam = st.sidebar.checkbox("Enable Webcam Input", value=False)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
show_history = st.sidebar.checkbox("Show Prediction History", value=True)

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Image input: File uploader or webcam
image = None
if use_webcam:
    webcam_image = st.camera_input("Capture Leaf Image from Webcam")
    if webcam_image is not None:
        image = Image.open(webcam_image).convert("RGB")
else:
    uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

if image is not None:
    # Display uploaded/captured image with futuristic border
    st.image(image, caption="Scanned Leaf Specimen", use_column_width=True)

    # Preprocess image
    img_size = (128, 128)  # Adjust if needed
    img_array = image.resize(img_size)
    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Simulate futuristic loading with spinner and progress bar
    with st.spinner("🔬 Analyzing with Neural Network..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Simulate processing time
            progress_bar.progress(i + 1)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = classes_list[predicted_index]
        confidence = float(np.max(predictions))

    # Check confidence threshold
    if confidence < confidence_threshold:
        st.warning(f"⚠️ Low confidence ({confidence:.2f}). Consider rescanning or uploading a clearer image.")
    else:
        # Show results with neon effects
        st.subheader("✅ Prediction Result")
        st.markdown(f"<h3 style='color: #00ff00;'>Class: {predicted_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #00ff00;'>Confidence: {confidence:.2f}</h3>", unsafe_allow_html=True)

        # Disease information
        st.subheader("📡 Disease Intelligence")
        info = disease_info.get(predicted_class, "No additional information available.")
        st.write(info)

        # Top-5 predictions with interactive bar chart (upgraded from top-3)
        st.subheader("🔎 Probability Spectrum")
        top_indices = predictions[0].argsort()[-5:][::-1]
        top_classes = [classes_list[i] for i in top_indices]
        top_probs = [predictions[0][i] for i in top_indices]
        df = pd.DataFrame({"Class": top_classes, "Probability": top_probs})
        fig = px.bar(df, x="Probability", y="Class", orientation='h', color="Probability",
                     color_continuous_scale="plasma")
        st.plotly_chart(fig, use_container_width=True)

        # Add to history
        st.session_state.prediction_history.append({
            "Class": predicted_class,
            "Confidence": confidence,
            "Image": image  # Store image for display (thumbnail)
        })

    # Prediction history in expander
    if show_history and st.session_state.prediction_history:
        with st.expander("📜 Mission Log (Prediction History)"):
            for idx, entry in enumerate(reversed(st.session_state.prediction_history)):
                st.write(f"Entry {len(st.session_state.prediction_history) - idx}:")
                st.image(entry["Image"].resize((100, 100)), caption=f"{entry['Class']} ({entry['Confidence']:.2f})")

# Footer
st.markdown("---")
st.write("Powered by xAI's Futuristic AI Engine | Version 2.0 | © 2025")
