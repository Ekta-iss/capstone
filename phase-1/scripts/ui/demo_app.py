import streamlit as st
import os
import time
import numpy as np
import sys
import glob
import cv2
import pickle
import pandas as pd
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

IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "combined", "video1", "images")
RADAR_PATH = os.path.join(PROJECT_ROOT, "data", "radar", "radar_features.pkl")
TCN_PATH = os.path.join(PROJECT_ROOT, "models", "tcn_radar.pt")

# =========================================================
# LOAD MODELS
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
# UI
# =========================================================
st.title("🚢 TERMINAL iQ — LIVE CRANE AI SYSTEM")

conf = st.slider("Detection Confidence", 0.1, 0.9, 0.4)
phase_threshold = st.slider("Phase Sensitivity", 1, 10, 3)
risk_threshold = st.slider("Risk Alert Threshold", 0.1, 1.0, 0.7)

col1, col2 = st.columns(2)

if not st.session_state.running:
    if col1.button("▶ START"):
        st.session_state.running = True

if st.session_state.running:
    if col2.button("⛔ STOP"):
        st.session_state.running = False

st.divider()

# =========================================================
# PLACEHOLDERS
# =========================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("📹 Live Computer Vision")
    cv_box = st.empty()

with right:
    st.subheader("📡 Radar Intelligence")
    radar_box = st.empty()

    st.subheader("🚨 System Status")
    alert_box = st.empty()

    m1, m2 = st.columns(2)
    phase_metric = m1.empty()
    risk_metric = m2.empty()

    st.write("**Optimization Output**")
    opt_box = st.empty()

# =========================================================
# LOGIC
# =========================================================
def detect_phase(n, th):
    if n == 0:
        return "IDLE"
    if n < th:
        return "ALIGNMENT"
    if n < th * 2:
        return "LIFT"
    return "TRANSPORT"

def classify_radar_state(risk):
    if risk < 0.3:
        return "SAFE", "🟢"
    elif risk < 0.7:
        return "WARNING", "🟡"
    else:
        return "CRITICAL", "🔴"

# =========================================================
# MAIN LOOP
# =========================================================
if st.session_state.running:

    for i in range(st.session_state.frame_idx, len(image_files)):

        if not st.session_state.running:
            break

        frame = cv2.imread(image_files[i])

        # ================= YOLO =================
        result = model.predict(frame, conf=conf, verbose=False)[0]

        dets = []
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            dets.append([x1, y1, x2, y2])

        # ================= TRACKING =================
        tracks = tracker.update(np.array(dets))

        # ================= RADAR SEQUENCE =================
        radar_seq = []
        for bbox, tid in tracks:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2 / frame.shape[1]
            cy = (y1 + y2) / 2 / frame.shape[0]
            radar_seq.append([cx, cy])

        radar_output = None

        # ================= TCN =================
        if tcn_model is not None and len(radar_seq) >= 5:
            seq = np.array(radar_seq[-10:])
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                radar_output = tcn_model(seq_tensor).cpu().numpy()

        if radar_output is None:
            radar_output = np.array([[np.mean(radar_seq) if radar_seq else 0]])

        # ================= CONTROL =================
        phase = detect_phase(len(tracks), phase_threshold)
        risk = float(np.clip(radar_output[0][0], 0, 1))

        control = optimize_control((0, 0, risk), risk)

        # ================= UI =================
        cv_box.image(result.plot(), channels="BGR", use_container_width=True)

        # 🔥 UPDATED RADAR PANEL
        state, icon = classify_radar_state(risk)

        with radar_box.container():
            st.metric("Risk Score", f"{risk:.2f}")

            if state == "SAFE":
                st.success(f"{icon} SAFE ZONE")
            elif state == "WARNING":
                st.warning(f"{icon} WARNING ZONE")
            else:
                st.error(f"{icon} CRITICAL ZONE")

            st.caption("AI-based spatial risk assessment")

        # METRICS
        phase_metric.metric("Phase", phase)
        risk_metric.metric("Risk", f"{risk:.2f}")

        # CONTROL OUTPUT
        opt_box.code(control)

        # ALERT
        if risk > risk_threshold:
            alert_box.error("⚠ HIGH RISK DETECTED")
        else:
            alert_box.success("SYSTEM SAFE")

        st.session_state.frame_idx += 1

        time.sleep(0.03)

else:
    cv_box.info("System Paused. Press START")