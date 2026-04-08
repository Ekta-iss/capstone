import os
import cv2
import numpy as np
import pickle

# ------------------------------
# Paths
# ------------------------------
IMAGES_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\images"
LABELS_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\labels"
OUTPUT_FILE = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"

# ------------------------------
# Helper functions
# ------------------------------
def load_labels(label_file):
    """
    Load YOLO format labels for a single image.
    Returns list of [class, x_center, y_center, width, height] in pixels
    """
    objs = []
    with open(label_file, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            objs.append([cls, x, y, w, h])
    return objs

# ------------------------------
# Main loop over frames
# ------------------------------
frame_files = sorted(os.listdir(IMAGES_FOLDER))
all_radar_features = []

prev_centers = {}  # track previous centers for velocity

for i, frame_file in enumerate(frame_files):
    img_path = os.path.join(IMAGES_FOLDER, frame_file)
    label_file = os.path.join(LABELS_FOLDER, frame_file.replace(".jpg", ".txt"))
    
    if not os.path.exists(label_file):
        all_radar_features.append([])  # no objects
        continue
    
    objs = load_labels(label_file)
    frame_features = []
    
    for obj in objs:
        cls, x, y, w, h = obj
        
        # Approx distance proxy: assume smaller boxes = farther
        size_proxy = w * h
        distance_proxy = 1.0 / (size_proxy + 1e-6)
        
        # Center in pixels
        center = np.array([x, y])
        
        # Velocity (frame-to-frame)
        key = f"{cls}_{x:.2f}_{y:.2f}"  # simple unique key per object
        if key in prev_centers:
            velocity = center - prev_centers[key]
        else:
            velocity = np.array([0.0, 0.0])
        
        prev_centers[key] = center.copy()
        
        frame_features.append({
            "class": cls,
            "center": center.tolist(),
            "velocity": velocity.tolist(),
            "distance": distance_proxy,
            "size": size_proxy
        })
    
    all_radar_features.append(frame_features)

# ------------------------------
# Save using pickle (works for variable-length frames)
# ------------------------------
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(all_radar_features, f)

print(f"✅ Radar-like features saved to {OUTPUT_FILE}")