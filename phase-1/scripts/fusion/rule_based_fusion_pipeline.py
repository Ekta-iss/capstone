import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import pickle

# =========================
# PATHS (FIXED & CLEAN)
# =========================
YOLO_MODEL_PATH = "../cv/runs/detect/runs/detect/yolov8_nano_fast/weights/best.pt"
RADAR_FILE = "../../data/radar/radar_features.pkl"
IMAGE_FOLDER = "../../data/combined/video1/images"

OUTPUT_PATH = "../../data/fusion/fusion_results.csv"

DEVICE = "cpu"


# =========================
# LOAD MODELS
# =========================
def load_models():
    print("🔗 Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL_PATH)
    return yolo


# =========================
# LOAD RADAR DATA
# =========================
def load_radar():
    with open(RADAR_FILE, "rb") as f:
        return pickle.load(f)


# =========================
# RISK ENGINE
# =========================
def compute_risk(distance, velocity, confidence, cls):

    risk = 0.0

    # distance risk
    if distance < 6:
        risk += 0.5
    elif distance < 12:
        risk += 0.2

    # velocity risk
    if abs(velocity) > 1.5:
        risk += 0.3

    # confidence risk
    if confidence < 0.5:
        risk += 0.2

    # class risk weighting
    if cls == 0:  # spreader
        risk += 0.3
    elif cls == 1:  # container
        risk += 0.1

    return min(risk, 1.0)


def safety_level(risk):
    if risk > 0.7:
        return "CRITICAL"
    elif risk > 0.4:
        return "WARNING"
    return "SAFE"


# =========================
# YOLO INFERENCE
# =========================
def run_yolo(yolo, image_path):

    results = yolo.predict(image_path, verbose=False)[0]

    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "class": cls,
            "confidence": conf,
            "cx": cx,
            "cy": cy
        })

    return detections


# =========================
# MAIN PIPELINE
# =========================
def main():

    print("🚀 Running RULE-BASED Fusion...")

    yolo = load_models()
    radar_data = load_radar()

    image_files = sorted(os.listdir(IMAGE_FOLDER))

    fusion_results = []

    for i, img_file in enumerate(image_files):

        img_path = os.path.join(IMAGE_FOLDER, img_file)

        # YOLO
        yolo_objects = run_yolo(yolo, img_path)

        # RADAR
        radar_frame = radar_data[i] if i < len(radar_data) else []

        for obj in yolo_objects:

            cls = obj["class"]
            conf = obj["confidence"]

            # simple association (can improve later with SORT/IoU tracking)
            if len(radar_frame) > 0:
                radar_obj = radar_frame[0]
                distance = radar_obj["distance"]
                velocity = radar_obj["velocity"]
            else:
                distance = 10.0
                velocity = 0.0

            # =========================
            # FUSION OUTPUT
            # =========================
            risk = compute_risk(distance, velocity, conf, cls)
            safety = safety_level(risk)

            fusion_results.append({
                "frame": img_file,
                "class": cls,
                "confidence": float(conf),
                "distance": float(distance),
                "velocity": float(velocity),
                "risk_score": float(risk),
                "safety": safety
            })

    # =========================
    # SAVE OUTPUT (CLEAN)
    # =========================
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df = pd.DataFrame(fusion_results)
    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Fusion completed successfully")
    print(f"📁 Saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()