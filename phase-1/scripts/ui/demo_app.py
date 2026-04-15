import streamlit as st
import os
import time
import numpy as np
import sys
import glob
import cv2
import pickle
import pandas as pd
from ultralytics import YOLO

# =========================================================
# ROOT (ROBUST)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
while os.path.basename(PROJECT_ROOT) != "scripts" and len(PROJECT_ROOT) > 3:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from scripts.fusion.control_optimizer import optimize_control

# =========================================================
# PATHS
# =========================================================
MODEL_GLOB = os.path.join(PROJECT_ROOT, "scripts", "cv", "runs", "**", "weights", "best.pt")
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "combined", "video1", "images")
RADAR_PATH = os.path.join(PROJECT_ROOT, "data", "radar", "radar_features.pkl")

# =========================================================
# CONFIG & STYLE
# =========================================================
st.set_page_config(page_title="Terminal iQ", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #38bdf8; }
    .topbar {
        background: linear-gradient(90deg,#0ea5e9,#1e3a8a);
        padding:15px;
        border-radius:10px;
        text-align:center;
        color: white;
        font-size:26px;
        font-weight:bold;
        margin-bottom:20px;
    }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #1e293b;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar'>🚢 TERMINAL iQ — LIVE CRANE AI SYSTEM</div>", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL & DATA
# =========================================================
@st.cache_resource
def load_yolo():
    matches = glob.glob(MODEL_GLOB, recursive=True)
    if not matches: return None
    return YOLO(matches[0])

@st.cache_data
def get_image_list():
    if not os.path.exists(IMAGE_FOLDER): return []
    return sorted([os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg", ".png"))])

@st.cache_data
def load_radar():
    if os.path.exists(RADAR_PATH):
        with open(RADAR_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_yolo()
image_files = get_image_list()
radar_data = load_radar()

if not model or not image_files:
    st.error("Missing Model Weights or Image Data. Check Paths.")
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    conf = st.slider("Detection Confidence", 0.1, 0.9, 0.4)
    phase_threshold = st.slider("Phase Sensitivity", 1, 10, 3)
    risk_threshold = st.slider("Risk Alert Threshold", 0.1, 1.0, 0.7)
    
    st.divider()
    
    c1, c2 = st.columns(2)
    if c1.button("▶ START", use_container_width=True):
        st.session_state.running = True
    if c2.button("⛔ STOP", use_container_width=True):
        st.session_state.running = False
        
    if st.button("🔄 RESET SYSTEM", use_container_width=True):
        st.session_state.frame_idx = 0
        st.session_state.running = False
        st.rerun()

# =========================================================
# UTILITIES
# =========================================================
def detect_phase(n, threshold):
    if n == 0: return "IDLE"
    if n < threshold: return "ALIGNMENT"
    if n < threshold * 2: return "LIFT"
    return "TRANSPORT"

def compute_risk(detections):
    if not detections: return 0.05
    cx = [d[0] for d in detections]
    spread = np.std(cx) if len(cx) > 1 else 0
    density = len(detections)
    return float(np.clip(0.1 * density + 0.4 * (spread/100), 0, 1))

# =========================================================
# MAIN DASHBOARD
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
    metric_box = st.container()

# =========================================================
# EXECUTION LOOP
# =========================================================
if st.session_state.running:
    # Use a local slice so session_state.frame_idx persists correctly
    current_batch = image_files[st.session_state.frame_idx:]
    
    for img_path in current_batch:
        if not st.session_state.running:
            break
            
        # 1. Processing
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        result = model.predict(frame, conf=conf, verbose=False)[0]
        plotted = result.plot()
        
        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            detections.append(((x1+x2)/2, (y1+y2)/2))

        # 2. Logic
        phase = detect_phase(len(detections), phase_threshold)
        risk = compute_risk(detections)
        control = optimize_control((0, 0, risk), risk)

        # 3. Updates
        cv_box.image(plotted, channels="BGR", use_container_width=True)
        
        # Radar Update
        if radar_data and st.session_state.frame_idx < len(radar_data):
            try:
                df_radar = pd.DataFrame(radar_data[st.session_state.frame_idx])
                radar_box.scatter_chart(
                    df_radar.rename(columns={"angle":"x","distance":"y"})[["x","y"]],
                    x="x", y="y", height=250
                )
            except:
                radar_box.write("Radar Sync Error")

        # Metrics Update
        with metric_box:
            m1, m2 = st.columns(2)
            m1.metric("Current Phase", phase)
            m2.metric("Risk Score", f"{risk:.2f}")
            st.write(f"**Optimization Output:** `{control}`")

        # Alert Logic
        if risk > risk_threshold:
            alert_box.error(f"⚠️ HIGH RISK DETECTED ({risk:.2f})")
        else:
            alert_box.success("✅ SYSTEM OPERATIONAL")

        st.session_state.frame_idx += 1
        time.sleep(0.01) # Reduced sleep for smoother playback

    if st.session_state.frame_idx >= len(image_files):
        st.session_state.running = False
        st.success("Sequence Complete.")
else:
    cv_box.info("System Paused. Press START to resume.")