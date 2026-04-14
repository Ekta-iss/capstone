import streamlit as st
import numpy as np
import json
import os
from PIL import Image
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = "../cv/runs/detect/runs/detect/yolov8_nano_fast/weights/best.pt"
IMAGE_FOLDER = "../../data/combined/video1/images"
KPI_PATH = "../../data/fusion/demo_kpi.json"

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO(YOLO_MODEL_PATH)

model = load_model()

# =========================
# LOAD KPI
# =========================
def load_kpi():
    if os.path.exists(KPI_PATH):
        with open(KPI_PATH, "r") as f:
            return json.load(f)
    return {}

# =========================
# UI
# =========================
st.title("🚢 Port Terminal AI Demo (Tuas Mega Port Use Case)")
st.write("YOLO + Radar + Fusion System")

kpi = load_kpi()

col1, col2, col3 = st.columns(3)

col1.metric("Throughput Cycles", kpi.get("throughput_cycles", 0))
col2.metric("Idle Time (sec)", kpi.get("idle_time_sec", 0))
col3.metric("Safety Events", kpi.get("safety_events", 0))

st.divider()

# =========================
# IMAGE VIEWER + YOLO
# =========================
images = sorted(os.listdir(IMAGE_FOLDER))
frame_idx = st.slider("Select Frame", 0, len(images)-1, 0)

img_path = os.path.join(IMAGE_FOLDER, images[frame_idx])
image = Image.open(img_path)

results = model.predict(img_path, verbose=False)[0]

annotated = results.plot()

st.image(annotated, caption=f"Frame {frame_idx}", use_container_width=True)

# =========================
# RAW DETECTIONS
# =========================
st.subheader("Detections")

for box in results.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    st.write(f"Class: {cls} | Confidence: {conf:.2f}")