import os
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
IMAGES_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\images"
LABELS_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\labels"
RADAR_PKL = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"
OUTPUT_FILE = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\merged_dataset.pkl"

CLASS_NAMES = {0: 'spreader', 1: 'container', 2: 'guide_mark', 3: 'target_slot'}
# BGR Colors: Blue, Green, Red, Yellow
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (0, 255, 255)}

# ------------------------------
# Helper Functions
# ------------------------------
def load_labels(label_file):
    """
    Converts YOLO center-format (x_c, y_c, w, h) to corner-format (x1, y1, x2, y2)
    to fix the box alignment drift.
    """
    objs = []
    if not os.path.exists(label_file):
        return objs
    with open(label_file, "r") as f:
        for line in f.readlines():
            cls, x_c, y_c, w, h = map(float, line.strip().split())
            
            # Math to shift from center to top-left corner
            x1 = x_c - (w / 2)
            y1 = y_c - (h / 2)
            x2 = x_c + (w / 2)
            y2 = y_c + (h / 2)
            
            objs.append({
                "class": int(cls),
                "bbox": [x1, y1, x2, y2]
            })
    return objs

def match_radar(cv_obj, radar_objects):
    cls = cv_obj["class"]
    candidates = [r for r in radar_objects if r["class"] == cls]
    if not candidates:
        return None
    return candidates[0]

def overlay_radar_on_image(img_path, merged_objects):
    """
    Visualizes fused data and handles keyboard interrupts.
    """
    img = cv2.imread(img_path)
    if img is None:
        return True
    H, W = img.shape[:2]

    for obj in merged_objects:
        # Scale corner coordinates to pixel values
        x1, y1, x2, y2 = [int(v*W) if i%2==0 else int(v*H) for i,v in enumerate(obj["bbox"])]
        cls = obj["class"]
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        
        # Draw Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Display Class and Radar Distance
        label = f"{obj['class_name']}: {obj['radar_distance']:.2f}m"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("SCASS Fused Output (Press 'q' to Quit)", img)
    
    # Check for 'q' or 'Esc' to close
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') or key == 27:
        return False
    return True

# ------------------------------
# Main Logic
# ------------------------------
with open(RADAR_PKL, "rb") as f:
    radar_data = pickle.load(f)

cv_images = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.endswith(".jpg")])
merged_dataset = []

num_frames = min(len(cv_images), len(radar_data))

for idx in tqdm(range(num_frames), desc="Merging frames"):
    img_file = cv_images[idx]
    img_path = os.path.join(IMAGES_FOLDER, img_file)
    label_file = os.path.join(LABELS_FOLDER, img_file.replace(".jpg", ".txt"))

    cv_objects = load_labels(label_file)
    radar_objects = radar_data[idx] # Now safe from IndexError

    merged_objects = []
    for cv_obj in cv_objects:
        radar_obj = match_radar(cv_obj, radar_objects)
        if radar_obj is not None:
            merged_objects.append({
                "bbox": cv_obj["bbox"],
                "class": cv_obj["class"],
                "class_name": CLASS_NAMES[cv_obj["class"]],
                "radar_distance": radar_obj["distance"],
                "radar_angle": radar_obj["angle"],
                "radar_track_id": radar_obj["track_id"]
            })

    merged_dataset.append({"image_path": img_path, "objects": merged_objects})

# Save merged dataset
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(merged_dataset, f)
print(f"✅ Merged dataset saved: {OUTPUT_FILE}")

# Quick Visual Inspection
print("\n--- Starting Visualization ---")
for frame in merged_dataset:
    if not overlay_radar_on_image(frame['image_path'], frame['objects']):
        print("Visualization ended by user.")
        break

cv2.destroyAllWindows()