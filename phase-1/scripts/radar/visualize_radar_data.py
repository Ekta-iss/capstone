import pickle
import matplotlib.pyplot as plt

# Image dimensions (replace with your actual video/image dimensions)
IMG_W, IMG_H = 1920, 1080  

RADAR_FILE = r"C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\radar\radar_features.pkl"

with open(RADAR_FILE, "rb") as f:
    radar_data = pickle.load(f)

# Visualize first N frames
N_FRAMES = 10

for frame_idx in range(N_FRAMES):
    frame_objects = radar_data[frame_idx]
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Radar-like frame {frame_idx}")
    plt.xlim(0, IMG_W)
    plt.ylim(0, IMG_H)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    
    for obj in frame_objects:
        # scale normalized center to pixel coords
        cx = obj['center'][0] * IMG_W
        cy = obj['center'][1] * IMG_H
        size = obj['size'] * max(IMG_W, IMG_H)  # scale size to image
        
        # Plot circle for object
        circle = plt.Circle((cx, cy), radius=size, fill=False, edgecolor='r', linewidth=2)
        plt.gca().add_patch(circle)
        
        # Plot center
        plt.plot(cx, cy, 'bo')
    
    plt.gca().invert_yaxis()  # optional: origin at top-left like images
    plt.show()