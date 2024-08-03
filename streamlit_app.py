import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL
import os

# Load the trained model
model_path = "grape_leaf_disease_3.0 .h5"  # Update this to your model's path

# Error handling for model loading
try:
    model = load_model(model_path)
    st.success('Model loaded successfully.')
except Exception as e:
    st.error(f'Error loading model: {e}')
    st.stop()  # Stop execution if model cannot be loaded

# Define categories
categories = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

# Streamlit app
st.title("Grape Disease Prediction")
st.write("Upload an image of a grape leaf to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Error handling for image processing
    try:
        # Load and preprocess the image
        image = PIL.Image.open(uploaded_file)
        image = image.resize((256, 256))
        img_array = np.array(image)
        
        # Check the shape of the image array
        st.write(f"Image shape: {img_array.shape}")
        
        # Ensure image is in the correct format (3 channels)
        if img_array.ndim == 2:  # If grayscale
            img_array = np.stack([img_array]*3, axis=-1)  # Convert to RGB
        elif img_array.shape[-1] != 3:  # If not RGB
            st.error('Image must have 3 channels (RGB).')
            st.stop()
        
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = categories[predicted_class]

        # Display result
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write(f'Prediction: {predicted_label}')
        st.write(f'Confidence: {predictions[0][predicted_class]:.2f}')
    except Exception as e:
        st.error(f'Error processing image: {e}')
