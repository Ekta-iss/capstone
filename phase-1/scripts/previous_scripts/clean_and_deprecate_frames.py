import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# --- Directory Configuration (Relative to scripts folder) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DATA_DIR = os.path.join(PHASE1_DIR, 'data')

# Source Folders
RAW_IMAGES_DIR = os.path.join(DATA_DIR, 'CVAT_annotated_Video1', 'obj_train_data')
LABEL_SOURCE_DIR = RAW_IMAGES_DIR # CVAT YOLO export puts .txt in same folder as .jpg

# Target Folder (Finalized for Training)
FINAL_TRAIN_DIR = os.path.join(DATA_DIR, 'clean_data', 'video1_clean_training')

def clean_watermark(frame):
    """
    Applies Navier-Stokes inpainting to the bottom-right corner.
    """
    h, w = frame.shape[:2]
    # User-provided relative coordinates
    y_start, y_end = int(h * 0.91), int(h * 0.98)
    x_start, x_end = int(w * 0.86), int(w * 0.98)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x_start-2, y_start-2), (x_end+2, y_end+2), 255, -1)
    
    # Inpaint and return
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

def main():
    print("--- Initializing AIS Data Pipeline ---")
    os.makedirs(FINAL_TRAIN_DIR, exist_ok=True)

    # 1. Get all image files
    all_files = os.listdir(RAW_IMAGES_DIR)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"Processing {len(image_files)} frames and syncing labels...")

    for filename in tqdm(image_files, desc="Sanitizing"):
        # Define paths
        img_path = os.path.join(RAW_IMAGES_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        label_name = base_name + ".txt"
        label_path = os.path.join(LABEL_SOURCE_DIR, label_name)

        # A. Process Image
        img = cv2.imread(img_path)
        if img is None: continue
        
        clean_img = clean_watermark(img)
        
        # Save image to final training folder
        cv2.imwrite(os.path.join(FINAL_TRAIN_DIR, filename), clean_img)

        # B. Process Label (Copy .txt if it exists)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(FINAL_TRAIN_DIR, label_name))

    # C. Copy metadata files for model training
    for meta_file in ['obj.names', 'obj.data']:
        meta_src = os.path.join(DATA_DIR, 'CVAT_annotated_Video1', meta_file)
        if os.path.exists(meta_src):
            shutil.copy(meta_src, os.path.join(FINAL_TRAIN_DIR, meta_file))

    print(f"\nSuccess! Cleaned data and synced labels are in: {FINAL_TRAIN_DIR}")

if __name__ == "__main__":
    main()