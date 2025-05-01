import streamlit as st
import gdown
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# ---------- 1. Download Model ----------
MODEL_URL = "https://drive.google.com/uc?id=16HM0INMWvC0yvSW9k-S_dauRRcvfgpyg"
MODEL_PATH = "best_model.pt"
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------- 2. Load YOLOv8 ----------
model = YOLO(MODEL_PATH)

# ---------- 3. Streamlit UI ----------
st.title("🧪 Bottle Cap Inspection with YOLOv8")
file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if file is not None:
    if file.type.startswith("image"):
        try:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            results = model.predict(source=np.array(img), save=False, conf=0.25)
            st.image(results[0].plot(), caption="Detection Result", use_column_width=True)
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
    
    elif file.type.startswith("video"):
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(file.read())
            temp_video_path = temp_video.name

        st.video(temp_video_path)
        st.info("🔍 Running detection on video...")
        results = model.predict(source=temp_video_path, save=True, conf=0.25)
        st.video(results[0].save_path)
