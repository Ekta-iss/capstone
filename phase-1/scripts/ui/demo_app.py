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

FOV_DEG = 60
MAX_RANGE = 20

# =========================================================
# LOAD MODELS (CACHED)
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
# LOAD DATA
# =========================================================
image_files = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
])

@st.cache_data
def load_radar():
    if os.path.exists(RADAR_PATH):
        with open(RADAR_PATH, "rb") as f:
            return pickle.load(f)
    return None

radar_data = load_radar()

# =========================================================
# SESSION STATE (NO FLICKER CONTROL)
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

if "last_radar" not in st.session_state:
    st.session_state.last_radar = None

if "last_control" not in st.session_state:
    st.session_state.last_control = None

# =========================================================
# UI (UNCHANGED STRUCTURE)
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
# PLACEHOLDERS (CRITICAL FOR NO FLICKER)
# =========================================================
left, right = st.columns([2, 1])

with left:
    st.subheader("📹 Live Computer Vision")
    cv_box = st.empty()

with right:
    st.subheader("📡 Radar Spatial Map")
    radar_box = st.empty()

    st.subheader("🚨 System Status")
    alert_box = st.empty()

    m1, m2 = st.columns(2)
    phase_metric = m1.empty()
    risk_metric = m2.empty()

    st.write("**Optimization Output**")
    opt_box = st.empty()

# =========================================================
# LOGIC FUNCTIONS
# =========================================================
def detect_phase(n, th):
    if n == 0:
        return "IDLE"
    if n < th:
        return "ALIGNMENT"
    if n < th * 2:
        return "LIFT"
    return "TRANSPORT"

def compute_risk(tracks):
    if len(tracks) == 0:
        return 0.05
    return float(np.clip(0.1 * len(tracks) + 0.3 * np.std([t[0] for t in tracks]), 0, 1))

# =========================================================
# MAIN LOOP (VIDEO STYLE EXECUTION)
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

        # =========================
        # SORT TRACKING
        # =========================
        tracks = tracker.update(np.array(dets))

        # =========================
        # RADAR SEQUENCE (FROM TRACKS)
        # =========================
        radar_seq = []
        for bbox, tid in tracks:
            x1, y1, x2, y2 = bbox

            cx = (x1 + x2) / 2 / frame.shape[1]
            cy = (y1 + y2) / 2 / frame.shape[0]

            radar_seq.append([cx, cy])

        radar_output = None

        # =========================
        # TCN RADAR INFERENCE (REAL MODEL OUTPUT)
        # =========================
        if tcn_model is not None and len(radar_seq) >= 5:
            seq = np.array(radar_seq[-10:])
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                radar_output = tcn_model(seq_tensor).cpu().numpy()

        # fallback (only if needed)
        if radar_output is None:
            radar_output = np.array([[np.mean(radar_seq) if radar_seq else 0]])

        # =========================
        # CONTROL INPUT
        # =========================
        phase = detect_phase(len(tracks), phase_threshold)
        risk = float(np.clip(radar_output[0][0], 0, 1))

        control = optimize_control((0, 0, risk), risk)

        # =========================
        # UI UPDATE (NO FLICKER)
        # =========================
        cv_box.image(result.plot(), channels="BGR", use_container_width=True)

        state = radar_state(float(risk))

        radar_box.markdown("### 📡 Radar Intelligence State")
        radar_box.markdown(f"## {state}")
        radar_box.metric("Raw Score", f"{risk:.2f}")

        phase_metric.metric("Phase", phase)
        risk_metric.metric("Risk", f"{risk:.2f}")

        opt_box.code(control)

        if risk > risk_threshold:
            alert_box.error("⚠ HIGH RISK DETECTED")
        else:
            alert_box.success("SYSTEM SAFE")

        st.session_state.frame_idx += 1

        time.sleep(0.03)

else:
    cv_box.info("System Paused. Press START")