import pickle
import random
from pprint import pprint

# ------------------------------
# Configuration
# ------------------------------
MERGED_PKL = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\merged_dataset.pkl"
OUTPUT_FOLDER = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\splits"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ------------------------------
# Load merged dataset
# ------------------------------
with open(MERGED_PKL, "rb") as f:
    merged_dataset = pickle.load(f)

# Shuffle dataset for randomness
random.shuffle(merged_dataset)

# ------------------------------
# Split indices
# ------------------------------
total = len(merged_dataset)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_set = merged_dataset[:train_end]
val_set = merged_dataset[train_end:val_end]
test_set = merged_dataset[val_end:]

# ------------------------------
# Save splits
# ------------------------------
import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(os.path.join(OUTPUT_FOLDER, "train.pkl"), "wb") as f:
    pickle.dump(train_set, f)
with open(os.path.join(OUTPUT_FOLDER, "val.pkl"), "wb") as f:
    pickle.dump(val_set, f)
with open(os.path.join(OUTPUT_FOLDER, "test.pkl"), "wb") as f:
    pickle.dump(test_set, f)

print("✅ Dataset splits saved:")
print(f"Train: {len(train_set)} frames")
print(f"Val:   {len(val_set)} frames")
print(f"Test:  {len(test_set)} frames")

# ------------------------------
# Preview first few frames
# ------------------------------
print("\n--- Sample from Train Set ---")
for frame in train_set[:3]:
    print(f"\nFrame: {frame['image_path']} | Objects: {len(frame['objects'])}")
    for obj in frame['objects']:
        print(f"  Class: {obj['class_name']}, Radar distance: {obj['radar_distance']:.2f}, BBox: {obj['bbox']}")