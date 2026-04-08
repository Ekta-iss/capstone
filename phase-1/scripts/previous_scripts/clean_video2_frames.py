import os
import cv2
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Update these paths to your local Video 2 directory
INPUT_DIR = r'C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\frames\video2_frames'
OUTPUT_DIR = r'C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\frames\video2_clean'

# Set to True to clean only every 5th frame for the PPT demo (saves time)
# Set to False to clean all 3000+ frames
SAMPLING_MODE = True 

def clean_frame(frame):
    h, w = frame.shape[:2]
    # Coordinates (Adjusted for bottom-right corner watermark)
    y_start, y_end = int(h * 0.91), int(h * 0.98)
    x_start, x_end = int(w * 0.86), int(w * 0.98)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x_start-2, y_start-2), (x_end+2, y_end+2), 255, -1)
    
    # Sharp inpainting for wharf texture
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))])
    
    if SAMPLING_MODE:
        files = files[::5] 
        print(f"Sampling Mode Active: Processing {len(files)} frames...")
    else:
        print(f"Full Clean Active: Processing {len(files)} frames...")

    for filename in tqdm(files, desc="Cleaning Video 2"):
        img = cv2.imread(os.path.join(INPUT_DIR, filename))
        if img is None: continue
        
        clean_img = clean_frame(img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), clean_img)

    print(f"\nSuccess! Cleaned frames saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()