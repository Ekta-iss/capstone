import pickle
import pandas as pd
import numpy as np

PKL_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"
CSV_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_telemetry2.csv"

CLASS_MAP = {
    0: "spreader",
    1: "container",
    2: "guide_mark",
    3: "target_slot"
}

def convert_pkl_to_csv():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    rows = []
    stats = {cls: 0 for cls in CLASS_MAP.keys()}

    for frame_idx, frame in enumerate(data):
        for obj in frame:
            # -------- ROBUST EXTRACTION --------
            class_id = obj.get("class", obj.get("class_id", -1))
            track_id = obj.get("track_id", "unknown")
            distance = obj.get("distance", 0)
            angle = obj.get("angle", 0)
            velocity = obj.get("velocity", 0)

            # -------- VALIDATE ANGLE --------
            if isinstance(angle, (list, tuple)):
                angle = angle[0]
            if abs(angle) > 180:  # Suspicious angle
                print(f"⚠️ WARNING: Frame {frame_idx}, Track {track_id}: Angle={angle}° (unusually large)")
                angle = np.clip(angle, -60, 60)  # Clip to FOV

            # -------- VALIDATE VELOCITY --------
            if isinstance(velocity, (list, tuple)):
                velocity = velocity[0]
            if abs(velocity) > 5:  # Suspicious velocity (>5 m/s)
                print(f"⚠️ WARNING: Frame {frame_idx}, Track {track_id}: Velocity={velocity} m/s (unusually high)")

            # -------- VALIDATE DISTANCE --------
            if distance < 0 or distance > 25:
                print(f"⚠️ WARNING: Frame {frame_idx}, Track {track_id}: Distance={distance}m (out of range)")

            # -------- CLASS VALIDATION --------
            class_name = CLASS_MAP.get(class_id, "unknown")
            if class_id not in CLASS_MAP:
                print(f"⚠️ WARNING: Frame {frame_idx}: Unknown class_id={class_id}")
            else:
                stats[class_id] += 1

            rows.append({
                "frame": frame_idx,
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "distance_m": round(distance, 4),
                "angle": round(angle, 4),
                "velocity": round(velocity, 4)
            })

    df = pd.DataFrame(rows)
    
    # -------- SAVE & REPORT --------
    df.to_csv(CSV_PATH, index=False)

    print("\n✅ CSV created successfully!")
    print(f"📊 Total detections: {len(df)}")
    print(f"📊 Unique frames: {df['frame'].nunique()}")
    print(f"📊 Unique tracks: {df['track_id'].nunique()}")
    print(f"\n📈 Class Distribution:")
    for cls_id, count in stats.items():
        print(f"   {CLASS_MAP[cls_id]}: {count} detections")
    print(f"\n📊 Statistics:")
    print(df[['distance_m', 'angle', 'velocity']].describe())

if __name__ == "__main__":
    convert_pkl_to_csv()