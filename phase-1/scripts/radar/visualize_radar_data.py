import numpy as np
import matplotlib.pyplot as plt
import pickle

RADAR_DATA_PATH = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"

def plot_scass_environment(radar_data, frame_idx=14):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location("N")
    
    # Color coding for SCASS classes
    colors = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}
    
    frame_objects = radar_data[frame_idx]
    
    for obj in frame_objects:
        cls_id = obj["class"]
        r = obj["distance"] * 20 
        theta = obj["angle"]
        
        ax.scatter(theta, r, c=colors[cls_id], s=250, label=obj["class_name"], edgecolors='black')
        ax.text(theta, r + 25, f"{obj['distance']:.2f}m", 
                color=colors[cls_id], fontweight='bold', ha='center')

    ax.set_ylim(0, 450)
    ax.set_title(f"SCASS Industrial Radar View - Frame {frame_idx}\n(Spreader, Container, Guide, Slot)", pad=35)
    
    # Legend cleanup (no duplicates)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.show()

if __name__ == "__main__":
    with open(RADAR_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    plot_scass_environment(data, frame_idx=14)