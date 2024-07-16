import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model with custom objects
model_path = "model/model.h5"
model = tf.keras.models.load_model(model_path)

# Labels and descriptions for prediction classes
Labels = ['Benign', 'Malignant', 'Non MRI']
Descriptions = {
    'Benign': 'No tumor detected.', 
    'Malignant': 'A malignant tumor is detected. Immediate medical attention is recommended.', 
    'Non MRI': 'Please add the brain MRI images if possible.'
}

# Function to preprocess image
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0  # Normalize pixel values
    return img

# Ensure the 'uploads' directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Streamlit app title and file uploader
st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image to a temporary location
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert the image to OpenCV format
    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Preprocess the image
    img = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))
    
    # Format prediction
    class_idx = np.argmax(prediction)
    class_label = Labels[class_idx]
    confidence = float(prediction[0][class_idx])
    
    # Get description for the predicted class
    description = Descriptions.get(class_label, 'No description available.')
    
    # Display the prediction and confidence
    st.write(f"Prediction: {class_label}")
    st.write(f"Confidence: {confidence:.2f}")
    st.write(f"Description: {description}")

    # HTML for result display
    st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f0f0;
    }
    .container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    h2 {
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-image {
        max-width: 100%;
        height: auto;
        border: 2px solid #333333;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .result-text {
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.write("<h2>Brain Tumor Classifier Result</h2>", unsafe_allow_html=True)
    st.image(image, caption='Uploaded Image.', use_column_width=True, output_format='PNG')
    st.write(f"<p><strong>Prediction:</strong> {class_label}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Confidence:</strong> {confidence:.2f}</p>", unsafe_allow_html=True)
    st.write(f"<p><strong>Description:</strong> {description}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
