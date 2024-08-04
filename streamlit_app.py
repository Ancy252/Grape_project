import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define a function to load the model and cache it
@st.cache_resource
def load_model_cached(model_path):
    try:
        model = load_model(model_path)
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the trained model
model_path = 'grape_leaf_disease_3.0.h5'
model = load_model_cached(model_path)

# Define categories
categories = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #8A2BE2, #FF69B4);
        padding: 10px;
        border-radius: 10px;
    }
    .title {
        font-size: 2em;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 1.5em;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 10px;
    }
    .file-uploader {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .uploaded-image {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .uploaded-image img {
        width: 80%;
        max-width: 300px;
        border: 5px solid #FFFFFF;
        border-radius: 50%; /* Circular shape */
    }
    .result-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .result-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #4B0082;
        background-color: #D3D3D3; /* Light gray background */
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .result-confidence {
        font-size: 1.5em;
        font-weight: bold;
        color: #2E8B57;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app
st.markdown('<div class="title">Grape Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a grape leaf to predict the disease.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key='file_uploader')

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = image.resize((256, 256))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Make prediction
        try:
            logging.info("Starting prediction...")
            predictions = model.predict(img_array)
            logging.info(f"Predictions: {predictions}")
            predicted_class = np.argmax(predictions[0])
            predicted_label = categories[predicted_class]
            confidence = predictions[0][predicted_class]

            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">Prediction: {predicted_label} 🎉</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-confidence">Confidence: {confidence:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except BrokenPipeError as e:
            logging.error(f"Broken pipe error during prediction: {e}")
            st.error(f"Error during prediction: {e}")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            st.error(f"Error during prediction: {e}")

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error(f"Error processing image: {e}")
        st.stop()
