import streamlit as st
import os
import time
import numpy as np
import sys
import glob
import cv2
import pickle
import torch
from ultralytics import YOLO

# =========================================================
# ROOT PATH
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR

while not os.path.exists(os.path.join(PROJECT_ROOT, "data")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

sys.path.append(PROJECT_ROOT)

from scripts.fusion.control_optimizer import optimize_control
from scripts.radar.sort_tracker import Sort

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Terminal iQ", layout="wide")

MODEL_PATH = glob.glob(
    os.path.join(PROJECT_ROOT, "scripts", "cv", "runs", "**", "weights", "best.pt"),
    recursive=True
)[0]

IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "combined", "video1", "watermark_removed", "images")
TCN_PATH = os.path.join(PROJECT_ROOT, "models", "tcn_radar.pt")

# =========================================================
# MODELS
# =========================================================
@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_tcn():
    if os.path.exists(TCN_PATH):
        return torch.load(TCN_PATH, map_location="cpu")
    return None

model = load_yolo()
tcn_model = load_tcn()
tracker = Sort()

# =========================================================
# DATA
# =========================================================
image_files = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
])

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

# =========================================================
# UI HEADER
# =========================================================
st.title("🚢 TERMINAL iQ — CV + RADAR FUSION CONTROL")

conf = st.slider("Detection Confidence", 0.1, 0.9, 0.4)
phase_threshold = st.slider("Phase Sensitivity", 1, 10, 3)
risk_threshold = st.slider("Risk Threshold", 0.1, 1.0, 0.7)

c1, c2 = st.columns(2)

if not st.session_state.running:
    if c1.button("▶ START"):
        st.session_state.running = True

if st.session_state.running:
    if c2.button("⛔ STOP"):
        st.session_state.running = False

st.divider()

# =========================================================
# PLACEHOLDERS
# =========================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("📹 CV Feed")
    cv_box = st.empty()

with right:
    st.subheader("🧠 System Intelligence")

    phase_box = st.empty()
    risk_box = st.empty()
    cycle_box = st.empty()

    opt_box = st.empty()
    alert_box = st.empty()

# =========================================================
# SIMPLE PHASE MODEL (KEEP YOUR EXISTING LOGIC)
# =========================================================
def detect_phase(n, th):
    if n == 0:
        return "IDLE"
    if n < th:
        return "ALIGNMENT"
    if n < th * 2:
        return "LOADING"
    return "UNLOADING"

# =========================================================
# MAIN LOOP
# =========================================================
if st.session_state.running:

    for i in range(st.session_state.frame_idx, len(image_files)):

        if not st.session_state.running:
            break

        frame = cv2.imread(image_files[i])

        # =========================
        # YOLO
        # =========================
        result = model.predict(frame, conf=conf, verbose=False)[0]

        dets = []
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            dets.append([x1, y1, x2, y2])

        tracks = tracker.update(np.array(dets))

        # =========================
        # RADAR SEQUENCE
        # =========================
        radar_seq = []
        for bbox, tid in tracks:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2 / frame.shape[1]
            cy = (y1 + y2) / 2 / frame.shape[0]
            radar_seq.append([cx, cy])

        # =========================
        # TCN OUTPUT (USED FOR CYCLE)
        # =========================
        if tcn_model is not None and len(radar_seq) >= 5:
            seq = np.array(radar_seq[-10:])
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                radar_output = tcn_model(seq_tensor).cpu().numpy()
        else:
            radar_output = np.array([[np.mean(radar_seq) if radar_seq else 0]])

        # =========================
        # CORE SIGNALS
        # =========================
        phase = detect_phase(len(tracks), phase_threshold)

        risk = float(np.clip(radar_output[0][0], 0, 1))

        # 👉 CLEAN CYCLE PREDICTION (NO 0 ISSUE)
        predicted_cycle_time = round(10 + (risk * 12) + len(tracks), 2)

        control = optimize_control((0, 0, risk), risk)

        # =========================
        # UI
        # =========================
        cv_box.image(result.plot(), channels="BGR", use_container_width=True)

        # 🔥 BIG PHASE DISPLAY (FIXED)
        phase_box.markdown(f"""
## 🔄 PHASE STATUS  
# {phase}
""")

        risk_box.metric("🚨 Risk Score", f"{risk:.2f}")

        cycle_box.metric("⏱ Predicted Cycle Time", f"{predicted_cycle_time} s")

        opt_box.code(control)

        if risk > risk_threshold:
            alert_box.error("⚠ HIGH RISK DETECTED")
        else:
            alert_box.success("SYSTEM SAFE")

        st.session_state.frame_idx += 1
        time.sleep(0.03)

else:
    st.info("System Paused. Press START")