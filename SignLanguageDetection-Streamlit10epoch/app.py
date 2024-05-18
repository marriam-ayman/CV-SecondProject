import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import tempfile

# YOLOv5 Model Loading (best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='he.pt')  # Replace 'best.pt' with your model's path

# Streamlit UI
st.title('YOLOv5 Object Detection')

# Input Options
upload_option = st.radio("Choose an input option:", ("Upload Image", "Real-Time Webcam"))

# Image Upload Handling
if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = model(image)  # Perform inference
        st.image(results.render()[0], caption='Detected Objects', use_column_width=True)  # Display results

# Real-Time Webcam Detection
if upload_option == "Real-Time Webcam":
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])  # Display window for webcam frames

    if run:
        cap = cv2.VideoCapture(0)  # Open webcam (0 for default)

        while run:
            ret, frame = cap.read()  # Capture frame
            if not ret:
                st.write("Error: Unable to capture frame")
                break
                
            # Convert to RGB and detect objects
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            annotated_frame = results.render()[0]
            FRAME_WINDOW.image(annotated_frame)

        cap.release()  # Release webcam