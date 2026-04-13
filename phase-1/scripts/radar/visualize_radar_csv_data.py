import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_telemetry.csv"

def plot_telemetry_trends():
    df = pd.read_csv(CSV_PATH)

    # Clean data
    df = df.dropna(subset=["frame", "distance_m", "class_name"])
    df = df.sort_values("frame")

    plt.figure(figsize=(14, 7))

    colors = {
        'spreader': '#1f77b4',
        'container': '#2ca02c',
        'guide_mark': '#ff7f0e',
        'target_slot': '#d62728'
    }

    # Stable ordering per class
    for name, group in df.groupby('class_name'):
        group = group.sort_values("frame")

        plt.plot(
            group['frame'],
            group['distance_m'],
            label=name.capitalize(),
            color=colors.get(name, 'gray'),
            linewidth=2,
            alpha=0.85
        )

    plt.title("SCASS Phase 1: Multi-Object Distance Telemetry", fontsize=16, pad=20)
    plt.xlabel("Frame Number", fontsize=12)
    plt.ylabel("Distance from Crane Origin (m)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Detected Objects", loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_telemetry_trends()