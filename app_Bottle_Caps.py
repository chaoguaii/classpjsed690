import streamlit as st
import gdown
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ---------- 1. Download Model from Google Drive ----------
MODEL_URL = "https://drive.google.com/uc?id=16HM0INMWvC0yvSW9k-S_dauRRcvfgpyg"
MODEL_PATH = "best_model.pt"

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------- 2. Load YOLOv8 Model ----------
model = YOLO(MODEL_PATH)

# ---------- 3. Streamlit UI ----------
st.title("🧪 Bottle Cap Inspection with YOLOv8")
st.markdown("Upload an image or video for defect detection.")

file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

if file is not None:
    file_bytes = file.read()
    file_type = file.type

    if file_type.startswith("image"):
        # ---------- 4.1 Inference on Image ----------
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        results = model.predict(source=np.array(img), save=False, conf=0.25)

        # Draw and display results
        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result", use_column_width=True)

    elif file_type.startswith("video"):
        # ---------- 4.2 Inference on Video ----------
        tfile = open("temp_video.mp4", "wb")
        tfile.write(file_bytes)
        tfile.close()

        st.info("⏳ Running detection on video...")
        results = model.predict(source="temp_video.mp4", save=True, conf=0.25)
        output_path = results[0].save_path

        st.video(output_path)
