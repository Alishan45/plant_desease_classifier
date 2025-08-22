import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load trained model
MODEL_PATH = "trained_plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class list
classes_list = ['Apple___Apple_scab',
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
 'Tomato___healthy']

# Disease information dictionary
disease_info = {
    'Apple___Apple_scab': {
        'description': 'Fungal disease causing olive-green to black velvety spots on leaves and fruit, leading to defoliation and fruit cracking.',
        'treatment': 'Apply fungicides like captan, plant resistant varieties, remove and destroy fallen leaves for sanitation.'
    },
    'Apple___Black_rot': {
        'description': 'Fungal disease with "frog-eye" leaf spots, fruit rot starting from blossom end, leading to mummified fruit.',
        'treatment': 'Prune infected branches, apply fungicides, maintain good sanitation.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Fungal disease causing yellow-orange spots on leaves and fruit, requires cedar as alternate host.',
        'treatment': 'Remove nearby cedar trees or galls, apply protective fungicides.'
    },
    'Apple___healthy': {
        'description': 'No disease detected. The apple plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Blueberry___healthy': {
        'description': 'No disease detected. The blueberry plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Fungal disease causing white powdery coating on leaves, leading to distorted growth and reduced yield.',
        'treatment': 'Improve air circulation, apply sulfur-based fungicides, remove infected parts.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'No disease detected. The cherry plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Fungal disease causing rectangular gray lesions with yellow halos on leaves.',
        'treatment': 'Use resistant hybrids, crop rotation, apply fungicides if necessary.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Fungal disease with reddish-brown pustules on both sides of leaves.',
        'treatment': 'Plant resistant varieties, apply fungicides in severe cases.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Fungal disease causing long, cigar-shaped tan lesions on leaves.',
        'treatment': 'Crop rotation, resistant hybrids, fungicides.'
    },
    'Corn_(maize)___healthy': {
        'description': 'No disease detected. The corn plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Grape___Black_rot': {
        'description': 'Fungal disease causing brown spots on leaves and black, shriveled berries.',
        'treatment': 'Sanitation by removing mummies, apply fungicides like mancozeb.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Fungal disease with tiger-striped leaves, wood necrosis, and sudden vine collapse.',
        'treatment': 'Prune out infected wood, no full cure, use preventive trunk injections.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Fungal disease causing angular brown spots on leaves.',
        'treatment': 'Apply fungicides, improve canopy management.'
    },
    'Grape___healthy': {
        'description': 'No disease detected. The grape plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Bacterial disease causing yellow shoots, mottled leaves, and small, misshapen fruit.',
        'treatment': 'Remove infected trees, control Asian citrus psyllid with insecticides.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial disease causing small spots on leaves leading to shot holes, and spots on fruit.',
        'treatment': 'Apply copper-based sprays, plant resistant varieties.'
    },
    'Peach___healthy': {
        'description': 'No disease detected. The peach plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial disease causing dark, water-soaked spots on leaves and fruit.',
        'treatment': 'Crop rotation, copper sprays, avoid overhead watering.'
    },
    'Pepper,_bell___healthy': {
        'description': 'No disease detected. The bell pepper plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Potato___Early_blight': {
        'description': 'Fungal disease with concentric ring spots on lower leaves.',
        'treatment': 'Crop rotation, fungicides like chlorothalonil, mulch.'
    },
    'Potato___Late_blight': {
        'description': 'Fungal disease causing water-soaked spots on leaves, white mold on undersides.',
        'treatment': 'Destroy infected plants, apply fungicides, use resistant varieties.'
    },
    'Potato___healthy': {
        'description': 'No disease detected. The potato plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Raspberry___healthy': {
        'description': 'No disease detected. The raspberry plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Soybean___healthy': {
        'description': 'No disease detected. The soybean plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Squash___Powdery_mildew': {
        'description': 'Fungal disease causing white powdery spots on leaves and stems.',
        'treatment': 'Apply fungicides, ensure good spacing for air flow, use resistant varieties.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Fungal disease causing purple spots and scorched leaf edges.',
        'treatment': 'Fungicides, remove infected leaves, improve air circulation.'
    },
    'Strawberry___healthy': {
        'description': 'No disease detected. The strawberry plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial disease causing small, dark spots on leaves and fruit.',
        'treatment': 'Copper-based bactericides, crop rotation, sanitation.'
    },
    'Tomato___Early_blight': {
        'description': 'Fungal disease with concentric rings on leaves starting from bottom.',
        'treatment': 'Fungicides, mulch, stake plants for air flow.'
    },
    'Tomato___Late_blight': {
        'description': 'Fungal disease with irregular water-soaked spots, white fuzzy growth.',
        'treatment': 'Remove infected parts, apply fungicides, avoid wet foliage.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Fungal disease with yellow spots on upper leaf surface, olive mold below.',
        'treatment': 'Improve ventilation, fungicides, resistant varieties.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Fungal disease with small circular spots with dark centers on leaves.',
        'treatment': 'Fungicides, remove lower leaves, mulch.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Pest causing stippling, yellowing, and fine webbing on leaves.',
        'treatment': 'Spray with water or miticides, introduce predatory mites.'
    },
    'Tomato___Target_Spot': {
        'description': 'Fungal disease with concentric rings on leaves and fruit.',
        'treatment': 'Fungicides, sanitation, crop rotation.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Viral disease causing upward curling and yellowing of leaves.',
        'treatment': 'Control whitefly vectors, remove infected plants, use resistant varieties.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Viral disease causing mottled green leaves and distorted growth.',
        'treatment': 'Sanitation, avoid handling plants when wet, resistant varieties.'
    },
    'Tomato___healthy': {
        'description': 'No disease detected. The tomato plant appears healthy.',
        'treatment': 'Maintain good cultural practices to prevent diseases.'
    }
}

# Futuristic CSS for a sci-fi look
st.markdown("""
    <style>
    /* Dark futuristic background */
    .stApp {
        background-color: #0a192f;
        color: #ccd6f6;
    }
    /* Neon glow for titles */
    h1, h2, h3 {
        color: #64ffda;
        text-shadow: 0 0 5px #64ffda, 0 0 10px #64ffda;
    }
    /* Button styling */
    .stButton > button {
        background-color: #0a192f;
        color: #64ffda;
        border: 2px solid #64ffda;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #64ffda;
        color: #0a192f;
        box-shadow: 0 0 10px #64ffda;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #172a46;
    }
    /* Image caption */
    .caption {
        color: #8892b0;
    }
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #64ffda;
    }
    </style>
""", unsafe_allow_html=True)

# Page config for wide layout
st.set_page_config(page_title="Futuristic Plant Disease Classifier", layout="wide")

# Sidebar for additional features
st.sidebar.title("About")
st.sidebar.markdown("**Author:** ALi SHan")
st.sidebar.markdown("[GitHub: Alishan45](https://github.com/Alishan45)")
st.sidebar.markdown("This app uses AI to detect plant diseases with a futuristic interface.")
st.sidebar.markdown("Features added: Webcam input, top 3 predictions, disease info, dark theme.")

# Main UI
st.title("🚀 Futuristic Plant Disease Classifier")
st.write("Upload or capture a plant leaf image to identify diseases using advanced AI.")

# Input selection
input_method = st.selectbox("Choose input method:", ["Upload Image", "Use Webcam"])

image = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif input_method == "Use Webcam":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(io.BytesIO(camera_file.getvalue()))

if image is not None:
    # Display uploaded/captured image
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction with spinner
    with st.spinner('Analyzing with AI...'):
        predictions = model.predict(img_array)
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[::-1][:3]
    top_classes = [classes_list[i] for i in top_indices]
    top_confidences = [predictions[0][i] * 100 for i in top_indices]

    # Show results
    st.subheader("AI Predictions:")
    for rank, (cls, conf) in enumerate(zip(top_classes, top_confidences), 1):
        st.write(f"**Rank {rank}:** {cls} - Confidence: {conf:.2f}%")
    
    # Primary prediction
    result_class = top_classes[0]
    info = disease_info.get(result_class, {'description': 'Unknown', 'treatment': 'N/A'})

    st.subheader("Disease Details:")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Recommended Treatment:** {info['treatment']}")

    # Download results
    result_text = f"Primary Prediction: {result_class}\nConfidence: {top_confidences[0]:.2f}%\nDescription: {info['description']}\nTreatment: {info['treatment']}"
    st.download_button("Download Results", result_text, file_name="prediction_results.txt")
