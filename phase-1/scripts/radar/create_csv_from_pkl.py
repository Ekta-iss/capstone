import pickle
import pandas as pd
import os

# --- Configuration ---
PKL_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"
CSV_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_telemetry.csv"

def convert_pkl_to_csv():
    if not os.path.exists(PKL_PATH):
        print(f"Error: {PKL_PATH} not found.")
        return

    with open(PKL_PATH, "rb") as f:
        radar_data = pickle.load(f)

    flattened_rows = []

    # Iterate through frames and objects to build a tabular list
    for frame_idx, frame_objects in enumerate(radar_data):
        for obj in frame_objects:
            row = {
                "frame": frame_idx,
                "class": obj["class"],
                "class_name": obj["class_name"],
                "distance_m": obj["distance"],
                "angle_rad": obj["angle"],
                "track_id": obj["track_id"]
            }
            flattened_rows.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(flattened_rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Successfully converted PKL to CSV: {CSV_PATH}")
    print(df.head()) # Preview first few rows

if __name__ == "__main__":
    convert_pkl_to_csv()