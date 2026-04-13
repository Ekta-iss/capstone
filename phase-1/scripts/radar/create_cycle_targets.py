import pickle
import os
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------
INPUT_PKL = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\merged_dataset.pkl"
OUTPUT_PKL = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\merged_with_targets.pkl"

FPS = 3  # your frame extraction rate

# ------------------------------
# Load dataset
# ------------------------------
with open(INPUT_PKL, "rb") as f:
    data = pickle.load(f)

# ------------------------------
# Detect cycles using spreader
# ------------------------------
cycles = []
current_cycle = []

prev_distance = None

for i, frame in enumerate(data):
    # find spreader
    spreader = None
    for obj in frame['objects']:
        if obj['class'] == 0:
            spreader = obj
            break

    if spreader is None:
        continue

    dist = spreader['radar_distance']

    # Detect new cycle (spreader reset upwards)
    if prev_distance is not None and dist > prev_distance + 2:
        if len(current_cycle) > 5:
            cycles.append(current_cycle)
        current_cycle = []

    current_cycle.append((i, dist))
    prev_distance = dist

# add last cycle
if current_cycle:
    cycles.append(current_cycle)

print(f"✅ Detected {len(cycles)} cycles")

# ------------------------------
# Assign remaining time
# ------------------------------
for cycle in tqdm(cycles):
    total_len = len(cycle)

    for idx, (frame_idx, _) in enumerate(cycle):
        remaining_frames = total_len - idx
        remaining_time = remaining_frames / FPS

        data[frame_idx]['target_time'] = remaining_time

# ------------------------------
# Save
# ------------------------------
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Saved dataset with targets: {OUTPUT_PKL}")