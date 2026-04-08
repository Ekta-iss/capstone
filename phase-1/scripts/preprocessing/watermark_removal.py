#!/usr/bin/env python3
"""
Standalone Watermark Removal Script
Handles:
1. CVAT-labeled YOLO dataset (video1/obj_train_data/images + labels)
2. Raw frames of video2

Output:
- Video1: data/processed/video1/watermark_removed/
    images/   -> processed images
    labels/   -> copy of original labels
- Video2: data/processed/video2/watermark_removed/
    images/   -> processed frames
"""

import cv2
import os
import shutil

# ------------------------------
# BASE DIR (repo root)
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# ------------------------------
# VIDEO CONFIG
# ------------------------------
VIDEO_SETTINGS = {
    1: {  # CVAT-labeled YOLO dataset
        "input_images": os.path.join(BASE_DIR, "data/cvat/video1/obj_train_data/images"),
        "input_labels": os.path.join(BASE_DIR, "data/cvat/video1/obj_train_data/labels"),
        "output_images": os.path.join(BASE_DIR, "data/processed/video1/watermark_removed/images"),
        "output_labels": os.path.join(BASE_DIR, "data/processed/video1/watermark_removed/labels"),
        "labeled": True
    },
    2: {  # Raw frames
        "input_images": os.path.join(BASE_DIR, "data/raw/frames/video2_frames"),
        "input_labels": None,
        "output_images": os.path.join(BASE_DIR, "data/processed/video2/watermark_removed/images"),
        "output_labels": None,
        "labeled": False
    }
}

# ------------------------------
# WATERMARK REMOVAL FUNCTION
# ------------------------------
def remove_watermark(frame, video_id):
    h, w = frame.shape[:2]

    if video_id == 1:
        x_start, x_end = int(w * 0.86), int(w * 0.98)
        y_start, y_end = int(h * 0.88), int(h * 0.98)
    elif video_id == 2:
        x_start, x_end = int(w * 0.82), int(w * 0.99)
        y_start, y_end = int(h * 0.88), int(h * 0.98)
    else:
        return frame

    frame[y_start:y_end, x_start:x_end] = 0
    return frame

# ------------------------------
# PROCESS FUNCTION
# ------------------------------
def process_video(video_id):
    config = VIDEO_SETTINGS[video_id]
    input_folder = config["input_images"]
    output_folder = config["output_images"]
    labeled = config["labeled"]
    
    # Output labels folder for CVAT-labeled data
    output_labels_folder = config.get("output_labels", None)
    if labeled and output_labels_folder:
        os.makedirs(output_labels_folder, exist_ok=True)
        # Copy labels folder
        shutil.copytree(config["input_labels"], output_labels_folder, dirs_exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Process images
    image_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(('.png','.jpg','.jpeg')))
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(input_folder, img_file)
        frame = cv2.imread(img_path)
        frame = remove_watermark(frame, video_id)

        output_img_path = os.path.join(output_folder, img_file)  # keep same filename
        cv2.imwrite(output_img_path, frame)

    print(f"[Watermark Removal] Video{video_id}: {len(image_files)} frames processed → {output_folder}")
    if labeled:
        print(f"[Watermark Removal] Labels copied → {output_labels_folder}")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    process_video(1)  # CVAT-labeled YOLO dataset
    process_video(2)  # Raw video2 frames