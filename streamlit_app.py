import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Define a function to load the model and cache it
@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

# Load the trained model
model_path = 'grape_leaf_disease_3.0.h5'
try:
    model = load_model_cached(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define categories
categories = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]

# Streamlit app
st.title("Grape Disease Prediction")
st.write("Upload an image of a grape leaf to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = image.resize((256, 256))
        image = ImageOps.expand(image, border=10, fill='black')  # Add a dark border
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write("")
        with col2:
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        with col3:
            st.write("")

        # Make prediction
        try:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = categories[predicted_class]

            # Display prediction results
            st.markdown(f"<h3 style='text-align: center;'>Prediction: {predicted_label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center;'>Confidence: {predictions[0][predicted_class]:.2f}</h4>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()
