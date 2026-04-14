import streamlit as st
import os
import time
import numpy as np
import sys
import glob
from ultralytics import YOLO

# =========================================================
# AUTO PROJECT ROOT (works from scripts/ui)
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = CURRENT_DIR
while not os.path.exists(os.path.join(PROJECT_ROOT, "data")):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# IMPORT CONTROL OPTIMIZER
# =========================================================
from scripts.fusion.control_optimizer import optimize_control

# =========================================================
# YOLO MODEL AUTO DETECTION
# =========================================================
MODEL_GLOB = os.path.join(
    PROJECT_ROOT,
    "scripts",
    "cv",
    "runs",
    "**",
    "weights",
    "best.pt"
)

matches = glob.glob(MODEL_GLOB, recursive=True)

if len(matches) == 0:
    st.error(f"No YOLO model found: {MODEL_GLOB}")
    st.stop()

YOLO_MODEL_PATH = matches[0]

# =========================================================
# DATASET PATH
# =========================================================
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "combined", "video1", "images")

if not os.path.exists(IMAGE_FOLDER):
    st.error(f"Image folder not found: {IMAGE_FOLDER}")
    st.stop()

image_files = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

if len(image_files) == 0:
    st.error("No images found in dataset")
    st.stop()

# =========================================================
# UI SETUP
# =========================================================
st.set_page_config(layout="wide")
st.title("🚢 Port Crane Control Room (AI Digital Twin)")
st.caption("YOLO + Risk Model + Cycle Time + Control Optimization")

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "kpi" not in st.session_state:
    st.session_state.kpi = {
        "cycles": 0,
        "idle_time": 0.0,
        "last_active": False
    }

FRAME_TIME = 0.1

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO(YOLO_MODEL_PATH)

model = load_model()

# =========================================================
# YOLO DETECTION
# =========================================================
def run_yolo(img_path, conf_threshold):
    results = model.predict(img_path, verbose=False)[0]

    detections = []

    for b in results.boxes:
        conf = float(b.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({"conf": conf, "cx": cx, "cy": cy})

    return results, detections

# =========================================================
# FAKE TEMPORAL MODEL
# =========================================================
def fake_lstm_prediction(detections):
    if len(detections) == 0:
        return (999, 0, 0.9)

    avg_x = np.mean([d["cx"] for d in detections])
    avg_y = np.mean([d["cy"] for d in detections])

    return (
        max(0, 100 - avg_x),
        avg_y / 100,
        min(1.0, len(detections) * 0.2)
    )

# =========================================================
# CYCLE TIME PREDICTION (NEW)
# =========================================================
def estimate_cycle_time(detections, risk_score, prev_active):

    base_time = 20.0

    density_factor = 1 + len(detections) * 0.15
    risk_factor = 1 + risk_score * 1.5
    activity_factor = 0.85 if prev_active else 1.0

    cycle_time = base_time * density_factor * risk_factor * activity_factor

    return round(cycle_time, 2)

# =========================================================
# KPI UPDATE
# =========================================================
def update_kpi(active):
    kpi = st.session_state.kpi

    if active and not kpi["last_active"]:
        kpi["cycles"] += 1

    if not active:
        kpi["idle_time"] += FRAME_TIME

    kpi["last_active"] = active

# =========================================================
# CONTROLS
# =========================================================
conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4)

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Simulation"):
        st.session_state.running = True

with col2:
    if st.button("⛔ Stop Simulation"):
        st.session_state.running = False

# =========================================================
# PLACEHOLDERS
# =========================================================
img_box = st.empty()
kpi_box = st.empty()
control_box = st.empty()
debug_box = st.empty()

# =========================================================
# MAIN LOOP
# =========================================================
if st.session_state.running:

    for i, img_path in enumerate(image_files):

        if not st.session_state.running:
            break

        results, detections = run_yolo(img_path, conf_threshold)

        active = len(detections) > 0

        lstm_pred = fake_lstm_prediction(detections)
        risk_score = lstm_pred[2]

        # =====================================================
        # CYCLE TIME + CONTROL OPTIMIZATION (NEW CORE)
        # =====================================================
        cycle_time = estimate_cycle_time(detections, risk_score, st.session_state.kpi["last_active"])

        control = optimize_control(
            lstm_pred,
            risk_score,
            cycle_time
        )

        update_kpi(active)

        # =====================================================
        # IMAGE PANEL
        # =====================================================
        with img_box.container():
            st.image(results.plot(), caption=f"Frame {i}", use_container_width=True)

        # =====================================================
        # KPI PANEL
        # =====================================================
        with kpi_box.container():
            c1, c2, c3, c4 = st.columns(4)
            kpi = st.session_state.kpi

            c1.metric("Cycles", kpi["cycles"])
            c2.metric("Idle Time (s)", round(kpi["idle_time"], 2))
            c3.metric("Risk Score", round(risk_score, 2))
            c4.metric("Cycle Time (s)", cycle_time)

        # =====================================================
        # CONTROL PANEL
        # =====================================================
        with control_box.container():
            st.subheader("🧠 Control Decision")

            st.write(f"**Action:** {control['action']}")
            st.write(f"**Speed:** {control['speed']}")
            st.write(f"**Cycle Time Est.:** {control['cycle_time']}")
            st.write(f"**Reason:** {control['reason']}")

            if control["action"] == "STOP":
                st.error("🛑 STOP COMMAND")
            elif control["action"] == "SLOW":
                st.warning("⚠️ SLOW MODE")
            else:
                st.success("🟢 NORMAL OPERATION")

        # =====================================================
        # DEBUG PANEL
        # =====================================================
        with debug_box.container():
            st.json({
                "detections": len(detections),
                "active": active,
                "cycle_time": cycle_time,
                "frame": os.path.basename(img_path)
            })

        time.sleep(0.03)

    st.session_state.running = False
    st.success("✅ Simulation Completed")