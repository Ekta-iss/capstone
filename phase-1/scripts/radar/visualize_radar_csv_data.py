import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_telemetry.csv"

def plot_telemetry_trends():
    # Load the data
    df = pd.read_csv(CSV_PATH)
    
    plt.figure(figsize=(14, 7))
    
    # Industrial color palette for SCASS classes
    colors = {
        'spreader': '#1f77b4',     # Deep Blue
        'container': '#2ca02c',    # Forest Green
        'guide_mark': '#ff7f0e',   # Industrial Orange
        'target_slot': '#d62728'   # Safety Red
    }

    # Group by class_name to plot individual lines
    for name, group in df.groupby('class_name'):
        plt.plot(
            group['frame'], 
            group['distance_m'], 
            label=name.capitalize(), 
            color=colors.get(name, 'gray'),
            linewidth=2,
            alpha=0.8
        )

    # Styling the plot for your MTech presentation
    plt.title("SCASS Phase 1: Multi-Object Distance Telemetry", fontsize=16, pad=20)
    plt.xlabel("Timeline (Frame Number)", fontsize=12)
    plt.ylabel("Distance from Crane Origin (Meters)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Detected Objects", loc='upper right', frameon=True)
    
    # Highlight the stabilized nature of the data
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_telemetry_trends()