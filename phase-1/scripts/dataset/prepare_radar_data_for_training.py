# scripts/dataset/prepare_radar_data_for_training.py

import os
import pickle
import numpy as np
from collections import defaultdict

# ========================
# CONFIG
# ========================
INPUT_PKL = "../../data/radar/radar_features.pkl"
OUTPUT_DIR = "../../data/radar/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


# ========================
# LOAD DATA
# ========================
def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# ========================
# FRAME → TRACK FORMAT
# ========================
def convert_to_tracks(data):
    """
    Convert frame-wise data into track-wise sequences
    """
    tracks = defaultdict(list)

    for t, frame in enumerate(data):
        for obj in frame:
            tid = obj["track_id"]

            tracks[tid].append({
                "t": t,
                "distance": obj["distance"],
                "angle": obj["angle"],
                "velocity": obj["velocity"]
            })

    return tracks


# ========================
# SORT TRACKS BY TIME
# ========================
def sort_tracks(tracks):
    for tid in tracks:
        tracks[tid] = sorted(tracks[tid], key=lambda x: x["t"])
    return tracks


# ========================
# SPLIT PER TRACK (TIME-BASED)
# ========================
def split_track(track):
    n = len(track)

    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    return (
        track[:train_end],
        track[train_end:val_end],
        track[val_end:]
    )


# ========================
# EXTRACT FEATURES
# ========================
def extract_features(track):
    return np.array([
        [d["distance"], d["angle"], d["velocity"]]
        for d in track
    ], dtype=np.float32)


# ========================
# CREATE SEQUENCES
# ========================
def create_sequences(features):
    X, y = [], []

    for i in range(len(features) - SEQ_LEN):
        X.append(features[i:i+SEQ_LEN])
        y.append(features[i+SEQ_LEN])

    return X, y


# ========================
# MAIN PIPELINE
# ========================
def main():
    print("Loading radar data...")
    data = load_data(INPUT_PKL)

    print("Converting to track-wise format...")
    tracks = convert_to_tracks(data)

    print(f"Total tracks: {len(tracks)}")

    tracks = sort_tracks(tracks)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    print("Processing each track...")

    for tid, track in tracks.items():
        if len(track) < SEQ_LEN + 1:
            continue  # skip short tracks

        train_t, val_t, test_t = split_track(track)

        for split, X_list, y_list in [
            (train_t, X_train, y_train),
            (val_t, X_val, y_val),
            (test_t, X_test, y_test),
        ]:
            if len(split) < SEQ_LEN + 1:
                continue

            features = extract_features(split)
            X_seq, y_seq = create_sequences(features)

            X_list.extend(X_seq)
            y_list.extend(y_seq)

    # Convert to numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print("Saving datasets...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print("\n✅ Done!")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")


if __name__ == "__main__":
    main()