import os
import pickle
from tqdm import tqdm
import cv2
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
IMAGES_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\images"
LABELS_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\labels"
RADAR_PKL = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"
OUTPUT_FILE = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\merged_dataset.pkl"

CLASS_NAMES = {0: 'spreader', 1: 'container', 2: 'guide_mark', 3: 'target_slot'}
CLASS_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (0, 255, 255)}

# ------------------------------
# Helper Functions
# ------------------------------
def load_labels(label_file):
    objs = []
    if not os.path.exists(label_file): return objs
    with open(label_file, "r") as f:
        for line in f.readlines():
            cls, x_c, y_c, w, h = map(float, line.strip().split())
            objs.append({
                "class": int(cls),
                "bbox": [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2] # [x1, y1, x2, y2]
            })
    return objs

def overlay_radar_on_image(img_path, merged_objects):
    img = cv2.imread(img_path)
    if img is None: return True
    H, W = img.shape[:2]

    for obj in merged_objects:
        x1, y1, x2, y2 = [int(v*W) if i%2==0 else int(v*H) for i,v in enumerate(obj["bbox"])]
        color = CLASS_COLORS.get(obj["class"], (255, 255, 255))
        
        # 1. Draw the Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # 2. Setup Label with background "Badge" for clarity
        label = f"{obj['class_name']}: {obj['radar_distance']:.2f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (l_w, l_h), _ = cv2.getTextSize(label, font, 0.6, 2)
        
        # Position label inside box if near top edge, else above
        t_y1, t_y2 = (y1 + 5, y1 + l_h + 15) if y1 < 30 else (y1 - l_h - 15, y1 - 5)
        text_y = y1 + l_h + 5 if y1 < 30 else y1 - 10

        cv2.rectangle(img, (x1, t_y1), (x1 + l_w, t_y2), color, -1)
        cv2.putText(img, label, (x1, text_y), font, 0.6, (255, 255, 255), 2)

    cv2.imshow("SCASS Fused Output", img)
    return False if cv2.waitKey(0) & 0xFF == ord('q') else True

# ------------------------------
# Main Logic with Landing Lock
# ------------------------------
with open(RADAR_PKL, "rb") as f:
    radar_data = pickle.load(f)

cv_images = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.endswith(".jpg")])
merged_dataset = []

for idx in tqdm(range(min(len(cv_images), len(radar_data)))):
    img_path = os.path.join(IMAGES_FOLDER, cv_images[idx])
    label_file = os.path.join(LABELS_FOLDER, cv_images[idx].replace(".jpg", ".txt"))
    
    cv_objects = load_labels(label_file)
    radar_objects = {r["class"]: r for r in radar_data[idx]}

    # Step A: Identify ground-level reference (the container being landed on)
    ground_dist = None
    container_y = None
    for obj in cv_objects:
        if obj["class"] == 1: # Container
            ground_dist = radar_objects.get(1, {}).get("distance")
            container_y = obj["bbox"][1] # Top edge of container

    # Step B: Apply "Landing Lock" logic
    merged_objects = []
    for cv_obj in cv_objects:
        cls = cv_obj["class"]
        r_obj = radar_objects.get(cls)
        
        if r_obj:
            current_dist = r_obj["distance"]
            
            # If this is the spreader and it is vertically near the container y-level
            if cls == 0 and container_y is not None and ground_dist is not None:
                spreader_y_bottom = cv_obj["bbox"][3]
                # If spreader bottom is within 10% of container top
                if abs(spreader_y_bottom - container_y) < 0.15:
                    current_dist = ground_dist # LOCK DISTANCE

            merged_objects.append({
                "bbox": cv_obj["bbox"],
                "class": cls,
                "class_name": CLASS_NAMES[cls],
                "radar_distance": current_dist,
                "radar_angle": r_obj["angle"],
                "radar_track_id": r_obj["track_id"]
            })

    merged_dataset.append({"image_path": img_path, "objects": merged_objects})

# Save and Visualize
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(merged_dataset, f)

print("\n--- Starting Final Visualization ---")
for frame in merged_dataset:
    if not overlay_radar_on_image(frame['image_path'], frame['objects']): break
cv2.destroyAllWindows()