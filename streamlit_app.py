import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Set up the page configuration
st.set_page_config(page_title="Grape Disease Prediction", page_icon=":grapes:", layout="centered")

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

# Apply custom CSS for background and styling
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to bottom, #3b0a45, #000000);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #3b0a45, #000000);
    }
    .stImage img {
        max-width: 80%;
        margin: 0 auto;
        display: block;
    }
    .prediction-box {
        border: 2px solid #6a1b9a;
        border-radius: 10px;
        padding: 10px;
        background-color: #6a1b9a;
        color: white;
        text-align: center;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Grape Disease Prediction")
st.write("Upload an image of a grape leaf to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = image.resize((128, 128))  # Reduce the image size
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Display the image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False)

        # Make prediction
        try:
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = categories[predicted_class]
            confidence = predictions[0][predicted_class]

            # Display results
            st.markdown(f"""
                <div class="prediction-box">
                    Prediction: {predicted_label} <br>
                    Confidence: {confidence:.2f}
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()
