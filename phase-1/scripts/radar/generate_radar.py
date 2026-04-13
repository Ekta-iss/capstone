import os
import numpy as np
import pickle
from scipy.signal import savgol_filter
from tqdm import tqdm

# --- Configuration ---
LABELS_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\labels"
OUTPUT_FILE = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"

CLASS_NAMES = {0: 'spreader', 1: 'container', 2: 'guide_mark', 3: 'target_slot'}
# MAX_RANGE represents the distance to the furthest visible ground point (~18-20m slant range)
MAX_RANGE = 20.0 

def main():
    label_files = sorted([f for f in os.listdir(LABELS_FOLDER) if f.endswith('.txt')])
    all_radar_features = []
    raw_data_history = {cls: [] for cls in CLASS_NAMES.keys()}

    print("--- Phase 1: Capturing Sensor-Centric Data ---")
    for file in tqdm(label_files):
        path = os.path.join(LABELS_FOLDER, file)
        frame_detections = {cls: None for cls in CLASS_NAMES.keys()}
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f.readlines():
                    try:
                        cls, x, y, w, h = map(float, line.split())
                        cls = int(cls)
                        if cls in CLASS_NAMES:
                            # We use the 'y' coordinate as a proxy for depth
                            frame_detections[cls] = y
                    except ValueError: continue
        
        for cls in CLASS_NAMES.keys():
            if frame_detections[cls] is not None:
                val = frame_detections[cls]
            elif raw_data_history[cls]:
                val = raw_data_history[cls][-1]
            else:
                # Spreader starts in the foreground (0.8), Others in background (0.2)
                val = 0.8 if cls == 0 else 0.2
            raw_data_history[cls].append(val)

    print("--- Phase 2: Smoothing ---")
    smoothed_data = {cls: savgol_filter(values, 15, 3) if len(values) > 15 else values 
                     for cls, values in raw_data_history.items()}

    print("--- Phase 3: Slant Range Calculation ---")
    for i in range(len(label_files)):
        frame_list = []
        for cls in CLASS_NAMES.keys():
            y_smooth = smoothed_data[cls][i]
            
            # NATURAL FLOW LOGIC (Sensor Co-located with Camera):
            # y=1.0 (Bottom) -> Object is right under the lens (Distance = Min)
            # y=0.0 (Top)    -> Object is at the horizon (Distance = Max)
            
            # Base distance calculation
            dist_m = MAX_RANGE * (1.0 - y_smooth)
            
            # Offset: Ensure Spreader isn't '0m' (it hangs ~2-3m below camera)
            if cls == 0: 
                dist_m += 2.5 
            
            # Limit the ground objects to a realistic pier distance (e.g. 14-16m)
            if cls != 0 and dist_m < 8.0:
                dist_m = 14.5 + (y_smooth * 0.5) # Ground marks stay far

            frame_list.append({
                "class": cls,
                "class_name": CLASS_NAMES[cls],
                "distance": float(dist_m),
                "angle": 0.0,
                "track_id": cls
            })
        all_radar_features.append(frame_list)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_radar_features, f)
    print(f"✅ Co-located Radar Data Generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()