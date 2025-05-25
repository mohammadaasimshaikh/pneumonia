import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import os

from util import classify  # Assuming this is a utility function you created for classification


# Load the model and class names once to avoid reloading on each page change
@st.cache_resource
def load_classifier():
    model = load_model('./model/pneumonia_classifier.h5')
    with open('./model/labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    return model, class_names


model, class_names = load_classifier()

# Function for classifying an image
def classify_image(image):
    class_name, conf_score = classify(image, model, class_names)
    return class_name, conf_score


# Sidebar navigation for the app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Predict", "Gallery"])


# Home Page
if page == "Home":
    st.title("Pneumonia Detection using Image Classification")
    st.write("""
    Welcome to the Pneumonia Detection App using Image Classification! 
    This app is built to classify chest X-ray images as either 'Normal' or 'Pneumonia' using a trained convolutional neural network.
    
    ### How it works:
    - **Upload**: Upload a chest X-ray image and the model will predict whether it is normal or has pneumonia.
    - **Accuracy**: The model provides a confidence score along with the classification.
    - **Gallery**: Explore some sample chest X-rays of normal and pneumonia cases.
    
    Please use the sidebar to navigate to different sections.
    """)

# Upload & Predict Page
elif page == "Upload & Predict":
    st.title("Upload and Predict Pneumonia")
    st.header("Please upload a chest X-ray image")

    # Upload file section
    file = st.file_uploader("", type=["jpeg", "jpg", "png"])

    # Display and classify image
    if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify the uploaded image
        class_name, conf_score = classify_image(image)

        # Show classification result
        st.write(f"## Prediction: {class_name}")
        st.write(f"### Confidence Score: {conf_score * 100:.2f}%")

# Gallery Page (You can name this section as "X-ray Comparisons" or "Image Samples")
elif page == "Gallery":
    st.title("X-ray Comparisons: Normal vs Pneumonia")

    st.write("""
    Below are sample images comparing normal chest X-rays with those showing signs of pneumonia.
    Use these samples for reference or educational purposes.
    """)

    # Create columns for Normal vs Pneumonia
    col1, col2 = st.columns(2)

    # Image paths - Replace these paths with actual local paths to images
    normal_images_path = "./chest_xray/test/normal1/"
    pneumonia_images_path = "./chest_xray/test/pneumonia1/"

    # Fetch image files from the directories
    normal_images = [f for f in os.listdir(normal_images_path) if f.endswith(('jpeg', 'jpg', 'png'))]
    pneumonia_images = [f for f in os.listdir(pneumonia_images_path) if f.endswith(('jpeg', 'jpg', 'png'))]

    # Display the images side by side
    col1.header("Normal X-rays")
    col2.header("Pneumonia X-rays")

    # Ensure the same number of images in both columns for comparison
    for i in range(min(len(normal_images), len(pneumonia_images))):
        with col1:
            normal_img = Image.open(os.path.join(normal_images_path, normal_images[i]))
            st.image(normal_img, caption=f"Normal {i + 1}", use_column_width=True)

        with col2:
            pneumonia_img = Image.open(os.path.join(pneumonia_images_path, pneumonia_images[i]))
            st.image(pneumonia_img, caption=f"Pneumonia {i + 1}", use_column_width=True)