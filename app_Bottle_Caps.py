import streamlit as st
import gdown
import os
import tempfile
import io
from PIL import Image
import numpy as np
import imghdr
from ultralytics import YOLO

# --- CONFIG ---
st.set_page_config(page_title="YOLOv8 Bottle Cap Detector", layout="centered")

# --- DOWNLOAD MODEL ---
MODEL_URL = "https://drive.google.com/uc?id=16HM0INMWvC0yvSW9k-S_dauRRcvfgpyg"
MODEL_PATH = "best_model.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# --- LOAD MODEL ---
model = YOLO(MODEL_PATH)

# --- UI ---
st.title("🧪 Bottle Cap Detection using YOLOv8")
st.write("Upload an image or video to detect defects or correct caps.")

uploaded_file = st.file_uploader("📂 Upload image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if uploaded_file.type.startswith("image"):
        if imghdr.what(None, h=file_bytes) is None:
            st.error("❌ This is not a valid image file.")
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                st.image(img, caption="📷 Uploaded Image", use_container_width=True)

                with st.spinner("🔍 Running detection..."):
                    results = model.predict(source=np.array(img), save=False, conf=0.25)
                    result_img = results[0].plot()

                st.image(result_img, caption="✅ Detection Result", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error opening image: {e}")

    elif uploaded_file.type.startswith("video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
            temp_vid.write(file_bytes)
            temp_video_path = temp_vid.name

        st.video(temp_video_path)
        st.info("🔍 Running detection on video...")
        with st.spinner("Processing..."):
            results = model.predict(source=temp_video_path, save=True, conf=0.25)

        st.video(results[0].save_path)
