import torch
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import time

# 1. SETUP PATHS
# Adjust these so Python can find your model definitions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'radar')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fusion')))

from train_tcn import TCN
from train_fusion_mlp_auto import DashboardMLP

# 2. CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ETA_SCALE = 50.0  # Must match the scale used in training
MODES = ["🟢 AUTO-PILOT", "🟡 ASSISTED CHECK", "🔴 MANUAL HANDOVER"]

# 3. LOAD ALL MODELS
print("📦 Initializing VisionSync AI Suite...")
yolo = YOLO("../cv/runs/detect/runs/detect/yolov8_nano_fast/weights/best.pt")

tcn = TCN(input_size=3, hidden=128).to(DEVICE)
tcn.load_state_dict(torch.load("../../models/tcn_radar_improved.pth", map_location=DEVICE))
tcn.eval()

mlp = DashboardMLP(input_size=6).to(DEVICE)
mlp.load_state_dict(torch.load("../../models/fusion_dashboard_mlp.pth", map_location=DEVICE))
mlp.eval()

# 4. DATA STREAM (Simulated from X_test)
X_test = np.load("../../data/radar/processed/X_test.npy")

def run_dashboard():
    print("🖥️ Dashboard Online. Starting Port Simulation...")
    time.sleep(2)

    with torch.no_grad():
        for i in range(len(X_test)):
            # A. Get TCN Prediction (Foresight)
            current_seq = torch.tensor(X_test[i:i+1], dtype=torch.float32).to(DEVICE)
            pred_telemetry = tcn(current_seq).cpu().numpy()[0]
            p_dist, p_angle, p_vel = pred_telemetry

            # B. Simulated YOLO Confidence (In real-time, this comes from yolo.predict)
            # We assume 0.95 for this demo sequence
            conf = 0.95 

            # C. MLP Decision (Fusion)
            # Feature: [Curr Dist, Curr Vel, Pred Dist, Pred Vel, CV Conf, Pred Angle]
            feat_vector = torch.tensor([[
                X_test[i][-1][0], X_test[i][-1][2], 
                p_dist, p_vel, conf, p_angle
            ]], dtype=torch.float32).to(DEVICE)
            
            pred_eta_scaled, mode_logits = mlp(feat_vector)
            
            # D. Rescale Outputs
            real_eta = max(0, pred_eta_scaled.item() * ETA_SCALE)
            mode_idx = torch.argmax(mode_logits, dim=1).item()

            # E. UI RENDERING (Console Dashboard)
            os.system('cls' if os.name == 'nt' else 'clear')
            print("="*50)
            print(f"      PORT - CRANE CONTROL UNIT      ")
            print("="*50)
            print(f" SYSTEM STATUS:  {MODES[mode_idx]}")
            print(f" OPERATOR ROLE:  {'Monitor AI' if mode_idx < 2 else 'TAKE CONTROL NOW'}")
            print("-"*50)
            print(f" 📍 SENSOR DATA (RADAR + TCN)")
            print(f"    - Vertical Dist:  {p_dist:6.2f} m")
            print(f"    - Descent Rate:   {p_vel:6.2f} m/s")
            print(f"    - Predicted Sway: {p_angle:6.2f}°")
            print("-"*50)
            print(f" ⏱️ PREDICTIVE ANALYTICS")
            print(f"    - Est. Cycle Time: {real_eta:6.1f} sec")
            print(f"    - Target Lock:     {'[ LOCKED ]' if p_dist < 15 else '[ SEEKING ]'}")
            print("="*50)
            
            # Control the refresh rate for the presentation
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard Terminated by Operator.")