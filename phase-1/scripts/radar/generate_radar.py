import os
import numpy as np
import pickle
from scipy.signal import savgol_filter
from tqdm import tqdm

# --- Configuration ---
LABELS_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\labels"
OUTPUT_FILE = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"

# Class Mapping
CLASS_NAMES = {0: 'spreader', 1: 'container', 2: 'guide_mark', 3: 'target_slot'}
MAX_ALT = 15.0 

def main():
    label_files = sorted([f for f in os.listdir(LABELS_FOLDER) if f.endswith('.txt')])
    all_radar_features = []
    
    # Storage for smoothing (keyed by class)
    raw_data_history = {cls: [] for cls in CLASS_NAMES.keys()}

    print("--- Phase 1: Parsing All Classes ---")
    for file in tqdm(label_files):
        path = os.path.join(LABELS_FOLDER, file)
        frame_detections = {cls: None for cls in CLASS_NAMES.keys()}
        
        with open(path, 'r') as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.split())
                cls = int(cls)
                if cls in CLASS_NAMES:
                    frame_detections[cls] = y
        
        # Fill missing detections with last known value or default
        for cls in CLASS_NAMES.keys():
            val = frame_detections[cls] if frame_detections[cls] is not None else (raw_data_history[cls][-1] if raw_data_history[cls] else 0.5)
            raw_data_history[cls].append(val)

    print("--- Phase 2: Signal Stabilization ---")
    smoothed_data = {}
    for cls, values in raw_data_history.items():
        if len(values) > 51:
            smoothed_data[cls] = savgol_filter(values, 51, 3)
        else:
            smoothed_data[cls] = values

    print("--- Phase 3: Finalizing Feature Set ---")
    for i in range(len(label_files)):
        frame_list = []
        for cls in CLASS_NAMES.keys():
            y_smooth = smoothed_data[cls][i]
            dist_m = MAX_ALT * (1.0 - y_smooth)
            
            frame_list.append({
                "class": cls,
                "class_name": CLASS_NAMES[cls],
                "distance": float(dist_m),
                "angle": 0.0 if cls == 0 else (0.1 if cls == 1 else -0.1), # Slight offset for visual clarity
                "track_id": cls
            })
        all_radar_features.append(frame_list)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_radar_features, f)
    print(f"✅ Multi-object radar data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()