import streamlit as st
import torch
import numpy as np
import time
import pandas as pd
import cv2
from ultralytics import YOLO
import sys
import os
from datetime import datetime, timedelta

# ==========================================
# 1. PATH RESOLUTION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

sys.path.append(os.path.join(PROJECT_ROOT, "scripts", "radar"))
sys.path.append(os.path.join(PROJECT_ROOT, "scripts", "fusion"))

from train_tcn import TCN
from train_fusion_mlp_auto import DashboardMLP

# ==========================================
# 2. PAGE CONFIG & NO-CUTOFF STYLING
# ==========================================
st.set_page_config(page_title="Terminal iQ | Smart Port", layout="wide", page_icon="🏗️")

st.markdown("""
    <style>
    /* Sidebar Visibility */
    [data-testid="stSidebar"] { background-color: #111827; color: white !important; }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
        color: white !important; font-size: 1.05rem !important; font-weight: 600 !important;
    }

    /* CRITICAL FIX: Prevent Metric Cutoff (...) */
    [data-testid="stMetricValue"] > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
        font-size: 1.6rem !important; /* Slightly smaller to fit 2 decimal places */
    }
    
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #e5e7eb !important;
        padding: 10px 5px !important; /* Reduced padding for more width */
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        min-width: 120px !important;
    }

    [data-testid="stMetricLabel"] { 
        color: #4b5563 !important; 
        font-size: 0.9rem !important; 
        font-weight: 700 !important;
        white-space: nowrap !important;
    }

    .status-header { font-weight: 800; font-size: 24px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Terminal iQ")
st.markdown("##### *Operational Control Interface | Port Automation*")

# ==========================================
# 3. ASSET LOADING
# ==========================================
@st.cache_resource
def load_assets():
    device = torch.device("cpu")
    yolo_path = os.path.join(PROJECT_ROOT, "scripts", "cv", "runs", "detect", "runs", "detect", "yolov8_nano_fast", "weights", "best.pt")
    tcn_path = os.path.join(PROJECT_ROOT, "models", "tcn_radar_improved.pth")
    mlp_path = os.path.join(PROJECT_ROOT, "models", "fusion_dashboard_mlp.pth")
    data_path = os.path.join(PROJECT_ROOT, "data", "radar", "processed", "X_test.npy")

    yolo_model = YOLO(yolo_path)
    tcn_model = TCN(input_size=3, hidden=128)
    tcn_model.load_state_dict(torch.load(tcn_path, map_location=device))
    tcn_model.eval()
    
    mlp_model = DashboardMLP(input_size=6)
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
    mlp_model.eval()
    
    return yolo_model, tcn_model, mlp_model, np.load(data_path)

yolo, tcn, mlp, X_test = load_assets()

# ==========================================
# 4. OPERATOR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Terminal iQ Control")

st.sidebar.subheader("Safety Parameters")
safety_buffer = st.sidebar.slider("Proximity Buffer (m)", 1.0, 10.0, 3.0)
sway_tol = st.sidebar.slider("Sway Tolerance (°)", 5, 30, 15)

st.sidebar.subheader("Vision Config")
vision_min_conf = st.sidebar.slider("Vision Confidence Cutoff", 0.1, 0.9, 0.45)

st.sidebar.divider()
sim_speed = st.sidebar.select_slider("Inference Latency", options=[0.01, 0.05, 0.1, 0.2], value=0.05)
run_btn = st.sidebar.button("ENGAGE SYSTEM", use_container_width=True, type="primary")

# ==========================================
# 5. DASHBOARD LAYOUT
# ==========================================
col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    st.subheader("📡 Live Perception Stream")
    video_placeholder = st.empty()
    st.subheader("📈 Predictive Trajectory")
    chart_placeholder = st.empty()

with col2:
    st.subheader("🛡️ AI Safety Protocol")
    status_placeholder = st.empty()
    st.divider()
    
    st.subheader("📊 Real-Time Telemetry")
    m1, m2 = st.columns(2)
    dist_metric = m1.empty()
    vel_metric = m2.empty()
    
    st.subheader("⏱️ Efficiency Metrics")
    em1, em2 = st.columns(2)
    eta_metric = em1.empty()
    cycle_metric = em2.empty()

# ==========================================
# 6. EXECUTION ENGINE
# ==========================================
video_file = os.path.join(PROJECT_ROOT, "data", "raw", "videos", "video_01.mp4")
ETA_SCALE = 50.0

if run_btn:
    cap = cv2.VideoCapture(video_file)
    history_dist = []
    
    for i in range(len(X_test)):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        with torch.no_grad():
            results = yolo.predict(frame, conf=0.3, verbose=False)
            annotated_frame = results[0].plot() 
            current_conf = results[0].boxes.conf.mean().item() if len(results[0].boxes) > 0 else 0.0

            seq = torch.tensor(X_test[i:i+1], dtype=torch.float32)
            p_dist, p_angle, p_vel = tcn(seq).numpy()[0]

            feat = torch.tensor([[X_test[i][-1][0], X_test[i][-1][2], p_dist, p_vel, current_conf, p_angle]])
            p_eta_scaled, p_mode_logits = mlp(feat)
            
            if p_dist < safety_buffer or current_conf < vision_min_conf or abs(p_angle) > sway_tol:
                mode_idx = 2 
            else:
                mode_idx = torch.argmax(p_mode_logits).item()
            
            real_eta = max(0, p_eta_scaled.item() * ETA_SCALE)
            completion_time = (datetime.now() + timedelta(seconds=real_eta)).strftime('%H:%M:%S')

            # RENDERING
            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            
            colors = ["#10b981", "#f59e0b", "#ef4444"]
            modes = ["🟢 AUTOPILOT", "🟡 ASSISTED", "🔴 MANUAL"]
            
            status_placeholder.markdown(f"""
                <div style="padding:20px; border-radius:15px; background-color:{colors[mode_idx]}; color:white; text-align:center;">
                    <span class="status-header">{modes[mode_idx]}</span>
                </div>
            """, unsafe_allow_html=True)

            # Metrics forced to 2 decimal places with overflow protection
            dist_metric.metric("ALTITUDE", f"{p_dist:.2f} m")
            vel_metric.metric("VELOCITY", f"{p_vel:.2f} m/s")
            eta_metric.metric("ETA LAND", f"{real_eta:.1f} s")
            cycle_metric.metric("COMPL.", completion_time)
            
            history_dist.append(p_dist)
            if len(history_dist) > 30: history_dist.pop(0)
            chart_placeholder.area_chart(pd.DataFrame(history_dist, columns=["Descent Prediction Path"]))

            time.sleep(sim_speed)
    cap.release()
else:
    st.info("System Standby. Click 'ENGAGE SYSTEM'.")