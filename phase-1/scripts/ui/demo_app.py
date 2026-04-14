import streamlit as st
import os
import time
import numpy as np
import sys
from ultralytics import YOLO

# =========================================================
# AUTO-DETECT PROJECT ROOT (works from scripts/ui)
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
# YOLO MODEL AUTO-DETECTION
# =========================================================
import glob

MODEL_GLOB = os.path.join(
    PROJECT_ROOT,
    "scripts",
    "cv",
    "runs",
    "**",
    "weights",
    "best.pt"
)

model_matches = glob.glob(MODEL_GLOB, recursive=True)

if len(model_matches) == 0:
    st.error(f"❌ No YOLO model found at: {MODEL_GLOB}")
    st.stop()

YOLO_MODEL_PATH = model_matches[0]

# =========================================================
# IMAGE DATASET PATH (FIXED)
# =========================================================
IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "combined",
    "video1",
    "images"
)

if not os.path.exists(IMAGE_FOLDER):
    st.error(f"❌ Image folder not found: {IMAGE_FOLDER}")
    st.stop()

image_files = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

if len(image_files) == 0:
    st.error("❌ No images found in dataset folder")
    st.stop()

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(layout="wide")
st.title("🚢 Port Crane Operator Control Dashboard")
st.caption("YOLO + AI Risk Model + Control Optimizer Simulation")

# =========================================================
# SESSION STATE
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "kpi" not in st.session_state:
    st.session_state.kpi = {
        "cycles": 0,
        "safety_events": 0,
        "idle_time": 0.0,
        "last_active": False
    }

# =========================================================
# LOAD YOLO MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO(YOLO_MODEL_PATH)

model = load_model()

# =========================================================
# YOLO FUNCTION
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

        detections.append({
            "conf": conf,
            "cx": cx,
            "cy": cy
        })

    return results, detections

# =========================================================
# FAKE TEMPORAL MODEL
# =========================================================
def fake_lstm_prediction(detections):
    if len(detections) == 0:
        return (999, 0, 0.9)

    avg_x = np.mean([d["cx"] for d in detections])
    avg_y = np.mean([d["cy"] for d in detections])

    future_dist = max(0, 100 - avg_x)
    angle = avg_y / 100
    risk = min(1.0, len(detections) * 0.2)

    return (future_dist, angle, risk)

# =========================================================
# KPI ENGINE
# =========================================================
FRAME_TIME = 0.1

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

        control = optimize_control(lstm_pred, risk_score)

        update_kpi(active)

        # =========================
        # IMAGE
        # =========================
        with img_box.container():
            st.image(results.plot(), caption=f"Frame {i}", use_container_width=True)

        # =========================
        # KPI
        # =========================
        with kpi_box.container():
            c1, c2, c3, c4 = st.columns(4)
            kpi = st.session_state.kpi

            c1.metric("Cycles", kpi["cycles"])
            c2.metric("Idle Time (s)", round(kpi["idle_time"], 2))
            c3.metric("Risk Score", round(risk_score, 2))
            c4.metric("Safety Events", kpi["safety_events"])

        # =========================
        # CONTROL
        # =========================
        with control_box.container():
            st.subheader("🧠 AI Control Decision")

            st.write(f"**Action:** {control['action']}")
            st.write(f"**Speed:** {control['speed']}")
            st.write(f"**Reason:** {control['reason']}")

            if control["action"] == "STOP":
                st.error("🛑 STOP COMMAND")
            elif control["action"] == "SLOW":
                st.warning("⚠️ SLOW MODE")
            else:
                st.success("🟢 NORMAL OPERATION")

        # =========================
        # DEBUG
        # =========================
        with debug_box.container():
            st.json({
                "detections": len(detections),
                "frame": os.path.basename(img_path),
                "model": YOLO_MODEL_PATH
            })

        time.sleep(0.03)

    st.session_state.running = False
    st.success("✅ Simulation Completed")