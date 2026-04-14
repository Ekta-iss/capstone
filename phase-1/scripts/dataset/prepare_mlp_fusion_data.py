import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO

# =========================
# PATHS
# =========================
YOLO_MODEL_PATH = "../cv/runs/detect/runs/detect/yolov8_nano_fast/weights/best.pt"
RADAR_FILE = "../../data/radar/radar_features.pkl"
IMAGE_FOLDER = "../../data/combined/video1/images"

OUTPUT_X = "../../data/fusion/mlp_X.npy"
OUTPUT_Y = "../../data/fusion/mlp_Y.npy"


# =========================
# LOAD RADAR
# =========================
def load_radar():
    with open(RADAR_FILE, "rb") as f:
        return pickle.load(f)


# =========================
# RISK LABEL (GROUND TRUTH FOR MLP)
# =========================
def compute_risk(distance, velocity, conf, cls):
    risk = 0.0

    if distance < 6:
        risk += 0.5
    elif distance < 12:
        risk += 0.2

    if abs(velocity) > 1.5:
        risk += 0.3

    if conf < 0.5:
        risk += 0.2

    return min(risk, 1.0)


# =========================
# YOLO INFERENCE
# =========================
def run_yolo(model, image_path):
    results = model.predict(image_path, verbose=False)[0]

    detections = []

    for b in results.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = b.xyxy[0].tolist()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "class": cls,
            "conf": conf,
            "cx": cx,
            "cy": cy
        })

    return detections


# =========================
# MAIN
# =========================
def main():

    print("🔗 Preparing MLP Fusion Dataset (YOLO + Radar)")

    yolo = YOLO(YOLO_MODEL_PATH)
    radar_data = load_radar()

    image_files = sorted(os.listdir(IMAGE_FOLDER))

    X, Y = [], []

    for i, img_file in enumerate(image_files):

        img_path = os.path.join(IMAGE_FOLDER, img_file)

        yolo_objs = run_yolo(yolo, img_path)
        radar_frame = radar_data[i] if i < len(radar_data) else []

        for obj in yolo_objs:

            cls = obj["class"]
            conf = obj["conf"]
            cx = obj["cx"]
            cy = obj["cy"]

            # fallback radar match (improve later with tracking)
            if len(radar_frame) > 0:
                r = radar_frame[0]
                dist = r["distance"]
                vel = r["velocity"]
                angle = r["angle"]
            else:
                dist = 10
                vel = 0
                angle = 0

            risk = compute_risk(dist, vel, conf, cls)

            # =========================
            # FEATURES (X)
            # =========================
            X.append([
                cls,
                conf,
                dist,
                angle,
                vel,
                cx,
                cy
            ])

            # =========================
            # TARGET (Y)
            # =========================
            Y.append([
                dist + vel * 0.5,   # future distance (proxy)
                angle,              # future angle (simplified)
                risk
            ])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, Y)

    print("✅ MLP Fusion dataset created")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)


if __name__ == "__main__":
    main()