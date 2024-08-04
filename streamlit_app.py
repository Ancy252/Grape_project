import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

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
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Make prediction
        try:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = categories[predicted_class]
            confidence = predictions[0][predicted_class]

            # Display only the prediction label and confidence
            st.write(f'Prediction: {predicted_label}')
            st.write(f'Confidence: {confidence:.2f}')
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()
