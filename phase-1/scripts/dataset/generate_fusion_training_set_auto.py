import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import pickle
import os
import sys

# ========================
# 1. PATH SETUP
# ========================
# Add the radar directory to the python path to find train_tcn
RADAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'radar'))
if RADAR_DIR not in sys.path:
    sys.path.append(RADAR_DIR)

from train_tcn import TCN, TCNBlock, Chomp1d

# ========================
# 2. MODEL LOADING
# ========================
print("🚀 Loading Vision and Telemetry models...")

yolo_model_path = "../cv/runs/detect/runs/detect/yolov8_nano_fast/weights/best.pt"
tcn_model_path = "../../models/tcn_radar_improved.pth"

yolo_model = YOLO(yolo_model_path)

# Initialize TCN (Needs to match the training architecture)
tcn_model = TCN(input_size=3, hidden=128)
tcn_model.load_state_dict(torch.load(tcn_model_path, map_location=torch.device('cpu')))
tcn_model.eval()

# ========================
# 3. DASHBOARD LOGIC
# ========================
def determine_control_mode(dist, conf, sway):
    """
    Simulates the Pilot-Decision logic:
    Mode 0: Fully Autonomous (AI)
    Mode 1: Assisted (Human monitoring)
    Mode 2: Manual Handover (Danger/Precision)
    """
    if dist < 3.0 or conf < 0.4 or abs(sway) > 15:
        return 2  # Manual Handover
    elif dist < 10.0:
        return 1  # Assisted Mode
    return 0     # Auto Mode

def generate_fusion_dataset():
    # Load your historical sequences
    data_path = "../../data/radar/processed/X_test.npy"
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        return

    X_telemetry = np.load(data_path) 
    
    fusion_X = []
    fusion_Y = []

    print(f"🔄 Processing {len(X_telemetry)} sequences for fusion...")

    with torch.no_grad():
        for i in range(len(X_telemetry)):
            # A. Get TCN Prediction (Future state)
            seq = torch.tensor(X_telemetry[i:i+1], dtype=torch.float32)
            pred_next_state = tcn_model(seq).numpy()[0]
            pred_dist, pred_angle, pred_vel = pred_next_state

            # B. Get Current CV Context (Fixed for dataset prep, dynamic in live inference)
            conf = 0.95 
            
            # C. Calculate Dashboard Metrics for labels
            # Estimated Time to Arrival (ETA)
            eta = pred_dist / (abs(pred_vel) + 1e-6)
            
            # Control Mode
            mode = determine_control_mode(pred_dist, conf, pred_angle)

            # Feature Vector for MLP Input:
            # [Current Dist, Current Vel, Pred Dist, Pred Vel, CV Conf, Pred Angle]
            feat_vector = [
                X_telemetry[i][-1][0], # current dist from sequence end
                X_telemetry[i][-1][2], # current velocity from sequence end
                pred_dist, 
                pred_vel, 
                conf, 
                pred_angle
            ]
            
            fusion_X.append(feat_vector)
            fusion_Y.append([eta, mode])

    # ========================
    # 4. EXPORT LOGIC
    # ========================
    # Corrected Indentation: This logic now runs regardless of folder existence
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fusion'))

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created new folder: {OUTPUT_DIR}")

    x_path = os.path.join(OUTPUT_DIR, "fusion_X_final_auto.npy")
    y_path = os.path.join(OUTPUT_DIR, "fusion_Y_final_auto.npy")

    np.save(x_path, np.array(fusion_X))
    np.save(y_path, np.array(fusion_Y))

    print(f"💾 Files saved successfully in: {OUTPUT_DIR}")
    print(f"   -> Features: fusion_X_final_auto.npy (Shape: {np.array(fusion_X).shape})")
    print(f"   -> Labels:   fusion_Y_final_auto.npy (Shape: {np.array(fusion_Y).shape})")
    print("✅ Dashboard Fusion Dataset Ready for MLP Training.")

if __name__ == "__main__":
    generate_fusion_dataset()