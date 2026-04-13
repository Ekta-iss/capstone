import os
import numpy as np
import pickle
from tqdm import tqdm

LABELS_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\combined\video1\labels"
OUTPUT_FILE = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"

MAX_RANGE = 20.0
FPS = 30
FOV_DEG = 60  # camera field of view assumption

# ---------- IOU ----------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# ---------- YOLO → BBOX ----------
def yolo_to_bbox(x, y, w, h):
    return [x - w/2, y - h/2, x + w/2, y + h/2]

# ---------- MATCH ----------
def match_detections(prev_tracks, detections, iou_thresh=0.3):
    assigned = {}
    used = set()

    for tid, (prev_cls, prev_box) in prev_tracks.items():
        best_iou = 0
        best_idx = -1

        for i, (cls, det_box) in enumerate(detections):
            if i in used:
                continue

            score = iou(prev_box, det_box)
            if score > best_iou:
                best_iou = score
                best_idx = i

        if best_iou > iou_thresh:
            assigned[tid] = detections[best_idx]
            used.add(best_idx)

    return assigned, used

# ---------- MAIN ----------
def main():
    label_files = sorted([f for f in os.listdir(LABELS_FOLDER) if f.endswith(".txt")])

    tracks = {}  # tid -> (cls, bbox)
    next_id = 0

    prev_positions = {}
    all_radar_features = []

    for file in tqdm(label_files):
        path = os.path.join(LABELS_FOLDER, file)
        detections = []

        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    bbox = yolo_to_bbox(x, y, w, h)
                    detections.append((int(cls), bbox))

        assigned, used = match_detections(tracks, detections)

        new_tracks = {}

        # matched
        for tid, (cls, bbox) in assigned.items():
            new_tracks[tid] = (cls, bbox)

        # unmatched → new tracks
        for i, (cls, bbox) in enumerate(detections):
            if i not in used:
                new_tracks[next_id] = (cls, bbox)
                next_id += 1

        tracks = new_tracks

        # -------- RADAR FEATURES --------
        frame_list = []

        for tid, (cls, bbox) in tracks.items():
            x1, y1, x2, y2 = bbox

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # stable mapping
            dist = MAX_RANGE * (1 - cy)
            dist = np.clip(dist, 0, MAX_RANGE)

            angle = (cx - 0.5) * FOV_DEG

            velocity = 0.0
            if tid in prev_positions:
                velocity = (cy - prev_positions[tid]) * FPS

            prev_positions[tid] = cy

            frame_list.append({
                "track_id": tid,
                "class": cls,
                "distance": float(dist),
                "angle": float(angle),
                "velocity": float(velocity)
            })

        all_radar_features.append(frame_list)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_radar_features, f)

    print(f"✅ Saved radar features: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()