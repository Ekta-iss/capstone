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
# ROOT & PATHS (Logic remains the same)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
while os.path.basename(PROJECT_ROOT) != "scripts" and len(PROJECT_ROOT) > 3:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from scripts.fusion.control_optimizer import optimize_control

MODEL_GLOB = os.path.join(PROJECT_ROOT, "scripts", "cv", "runs", "**", "weights", "best.pt")
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "combined", "video1", "images")
RADAR_PATH = os.path.join(PROJECT_ROOT, "data", "radar", "radar_features.pkl")

# =========================================================
# CONFIG & HYBRID THEME STYLE
# =========================================================
st.set_page_config(page_title="Terminal iQ", layout="wide")

st.markdown("""
<style>
    /* 1. Main Dashboard (Dark) */
    .stApp { background-color: #0f172a; }
    h1, h2, h3, [data-testid="stMetricLabel"], .main .stMarkdown p {
        color: #ffffff !important;
    }

    /* 2. Left Panel / Sidebar (White Background, Black Text) */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Target all text, headers, and labels inside the sidebar */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        color: #000000 !important;
    }

    /* 3. Top Bar Styling */
    .topbar {
        background: linear-gradient(90deg,#0ea5e9,#1e3a8a);
        padding:15px;
        border-radius:10px;
        text-align:center;
        color: white !important;
        font-size:26px;
        font-weight:bold;
        margin-bottom:20px;
    }

    /* 4. Dashboard Specific Elements */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 32px;
    }
    code {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='topbar'>🚢 TERMINAL iQ — LIVE CRANE AI SYSTEM</div>", unsafe_allow_html=True)

# =========================================================
# DATA LOADING (Logic remains same)
# =========================================================
@st.cache_resource
def load_yolo():
    matches = glob.glob(MODEL_GLOB, recursive=True)
    return YOLO(matches[0]) if matches else None

@st.cache_data
def get_image_list():
    return sorted([os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg", ".png"))]) if os.path.exists(IMAGE_FOLDER) else []

@st.cache_data
def load_radar():
    if os.path.exists(RADAR_PATH):
        with open(RADAR_PATH, "rb") as f: return pickle.load(f)
    return None

model = load_yolo()
image_files = get_image_list()
radar_data = load_radar()

if not model or not image_files:
    st.error("Data or Model weights not found.")
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state: st.session_state.running = False
if "frame_idx" not in st.session_state: st.session_state.frame_idx = 0

# =========================================================
# SIDEBAR (WHITE BG / BLACK TEXT)
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
# MAIN DASHBOARD LAYOUT (DARK BG / WHITE TEXT)
# =========================================================
def detect_phase(n, threshold):
    if n == 0: return "IDLE"
    return "ALIGNMENT" if n < threshold else "LIFT" if n < threshold * 2 else "TRANSPORT"

def compute_risk(detections):
    if not detections: return 0.05
    spread = np.std([d[0] for d in detections]) if len(detections) > 1 else 0
    return float(np.clip(0.1 * len(detections) + 0.4 * (spread/100), 0, 1))

left, right = st.columns([2, 1])

with left:
    st.subheader("📹 Live Computer Vision")
    cv_box = st.empty()

with right:
    st.subheader("📡 Radar Spatial Map")
    radar_box = st.empty()
    st.divider()
    st.subheader("🚨 System Status")
    alert_box = st.empty()
    
    m_col1, m_col2 = st.columns(2)
    phase_metric = m_col1.empty()
    risk_metric = m_col2.empty()
    
    st.write("**Optimization Output:**")
    opt_box = st.empty()

# =========================================================
# EXECUTION LOOP
# =========================================================
if st.session_state.running:
    for img_path in image_files[st.session_state.frame_idx:]:
        if not st.session_state.running: break
            
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        result = model.predict(frame, conf=conf, verbose=False)[0]
        plotted = result.plot()
        
        detections = [((b.xyxy[0][0]+b.xyxy[0][2])/2, (b.xyxy[0][1]+b.xyxy[0][3])/2) for b in result.boxes]
        phase = detect_phase(len(detections), phase_threshold)
        risk = compute_risk(detections)
        control = optimize_control((0, 0, risk), risk)

        cv_box.image(plotted, channels="BGR", use_container_width=True)
        
        if radar_data and st.session_state.frame_idx < len(radar_data):
            try:
                df_radar = pd.DataFrame(radar_data[st.session_state.frame_idx])
                radar_box.scatter_chart(df_radar.rename(columns={"angle":"x","distance":"y"})[["x","y"]], x="x", y="y", height=250)
            except: pass

        phase_metric.metric("Current Phase", phase)
        risk_metric.metric("Risk Score", f"{risk:.2f}")
        opt_box.code(f"{control}")

        if risk > risk_threshold: alert_box.error("⚠️ HIGH RISK DETECTED")
        else: alert_box.success("✅ SYSTEM SAFE")

        st.session_state.frame_idx += 1
        time.sleep(0.01)
else:
    cv_box.info("System Paused. Press START to resume.")