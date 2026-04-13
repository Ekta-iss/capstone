import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

RADAR_DATA_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"
OUTPUT_DIR = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- CLASS MAP ----------
COLORS = {
    0: "blue",
    1: "green",
    2: "orange",
    3: "red"
}

CLASS_NAMES = {
    0: "Spreader",
    1: "Container",
    2: "Guide Mark",
    3: "Target Slot"
}

# ---------- DEBUG ----------
def inspect_pickle(data):
    print("\n=== DEBUG INFO ===")
    print(f"Total frames: {len(data)}")
    if len(data) > 14 and data[14]:
        print(f"Frame 14 count: {len(data[14])}")
        print(f"Sample object: {data[14][0]}")
    print("==================\n")

# ---------- POLAR VIEW ----------
def plot_polar(data, frame_idx=0, save=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("N")

    frame = data[frame_idx]
    seen = set()

    for obj in frame:
        cls = obj["class"]
        r = obj["distance"]
        theta = np.deg2rad(obj["angle"])

        label = CLASS_NAMES.get(cls, str(cls)) if cls not in seen else None
        seen.add(cls)

        ax.scatter(theta, r,
                   c=COLORS.get(cls, "gray"),
                   s=200,
                   edgecolors="black",
                   label=label)

        ax.text(theta, r + 0.5, f"{r:.1f}m",
                ha="center", fontsize=8)

    ax.set_title(f"Radar Polar View - Frame {frame_idx}")
    ax.set_ylim(0, 20)
    ax.legend()

    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, f"polar_{frame_idx}.png"), dpi=150)

    plt.show()
    plt.close()

# ---------- CARTESIAN VIEW ----------
def plot_cartesian(data, frame_idx=0, save=False):
    fig, ax = plt.subplots(figsize=(10, 10))

    frame = data[frame_idx]
    seen = set()

    for obj in frame:
        cls = obj["class"]
        dist = obj["distance"]
        angle = obj["angle"]
        vel = obj["velocity"]
        tid = obj["track_id"]

        theta = np.deg2rad(angle)

        x = dist * np.sin(theta)
        y = dist * np.cos(theta)

        label = CLASS_NAMES.get(cls, str(cls)) if cls not in seen else None
        seen.add(cls)

        ax.scatter(x, y,
                   c=COLORS.get(cls, "gray"),
                   s=250,
                   edgecolors="black",
                   label=label)

        if vel > 0:
            ax.arrow(x, y, 0, vel * 0.5,
                     head_width=0.3,
                     color=COLORS.get(cls, "gray"))

        ax.text(x, y + 0.3, f"ID {tid}", fontsize=8)

    ax.set_title(f"Radar Cartesian View - Frame {frame_idx}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 20)
    ax.grid(True)

    ax.legend()

    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, f"cartesian_{frame_idx}.png"), dpi=150)

    plt.show()
    plt.close()

# ---------- MAIN ----------
if __name__ == "__main__":
    with open(RADAR_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    inspect_pickle(data)

    print("Showing Frame 14...\n")

    plot_polar(data, frame_idx=14)
    plot_cartesian(data, frame_idx=14)