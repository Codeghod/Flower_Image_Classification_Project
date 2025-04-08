import os
import tensorflow as tf
import numpy as np
import keras
from keras.models import load_model
import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Flower Classification"])

if page == "Home":
    st.title("Welcome to the Flower Classification App")
    st.write("This app allows you to classify flower images using a pre-trained model.")
    st.write("You can upload an image of a flower, and the app will predict the type of flower.")
    st.write("The types of flowers that can be classified are:")
    st.write("- Daisy")
    st.write("- Lavender")
    st.write("- Lily")
    st.write("- Rose")
    st.write("- Sunflower")
    st.write("Use the navigation menu to go to the Flower Classification page.")
else:
    st.header('Flower Classification Model')
    flower_names = ['Daisy', 'Lavender', 'Lily', 'Rose', 'Sunflower']

    # Check if the model file exists
    model_path = 'flower_classification_model_2.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure 'flower_classification_model_2.h5' is in the directory.")
    else:
        model = load_model(model_path)

    def classify_images(image_path):
        try:
            input_image = tf.keras.utils.load_img(image_path, target_size=(256, 256))
            input_image_array = tf.keras.utils.img_to_array(input_image)
            input_image_exp_dim = tf.expand_dims(input_image_array, axis=0)

            predictions = model.predict(input_image_exp_dim)
            result = tf.nn.softmax(predictions[0])
            outcome = 'The Image belongs to' + ' ' + flower_names[np.argmax(result)]
            return outcome
        except Exception as e:
            return f"Error in image classification: {str(e)}"

    # Ensure the upload directory exists
    upload_dir = 'upload'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    uploaded_file = st.file_uploader('Upload an Image')
    if uploaded_file is not None:
        with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, width=200)
        # Call classify_images only if a valid image is uploaded
        result = classify_images(os.path.join(upload_dir, uploaded_file.name))
        st.markdown(result)
