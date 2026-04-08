#!/usr/bin/env python3
"""
Standalone Resize Script
Resizes denoised images for YOLO training.
Handles:
1. Video1 (CVAT-labeled)
   - Images: denoised/
   - Labels: watermark_removed/labels/
2. Video2 (raw frames)
   - Images: denoised/
   - No labels
Outputs:
- video1: data/processed/video1/resized/
    images/   -> resized images
    labels/   -> copied labels
- video2: data/processed/video2/resized/
    images/   -> resized frames
"""

import cv2
import os
import shutil

# ------------------------------
# YOLO TARGET SIZE
# ------------------------------
TARGET_WIDTH = 640
TARGET_HEIGHT = 640

# ------------------------------
# BASE DIR
# ------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# ------------------------------
# VIDEO SETTINGS
# ------------------------------
VIDEO_SETTINGS = {
    1: {  # Video1: CVAT-labeled
        "input_images": os.path.join(BASE_DIR, "data/processed/video1/denoised"),
        "input_labels": os.path.join(BASE_DIR, "data/processed/video1/watermark_removed/labels"),
        "output_images": os.path.join(BASE_DIR, "data/processed/video1/resized/images"),
        "output_labels": os.path.join(BASE_DIR, "data/processed/video1/resized/labels"),
        "labeled": True
    },
    2: {  # Video2: raw frames
        "input_images": os.path.join(BASE_DIR, "data/processed/video2/denoised"),
        "input_labels": None,
        "output_images": os.path.join(BASE_DIR, "data/processed/video2/resized/images"),
        "output_labels": None,
        "labeled": False
    }
}

# ------------------------------
# PROCESS FUNCTION
# ------------------------------
def resize_images(video_id):
    config = VIDEO_SETTINGS[video_id]
    input_folder = config["input_images"]
    output_folder = config["output_images"]
    labeled = config["labeled"]

    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Copy labels for CVAT-labeled dataset
    if labeled and config["input_labels"] and config["output_labels"]:
        os.makedirs(config["output_labels"], exist_ok=True)
        shutil.copytree(config["input_labels"], config["output_labels"], dirs_exist_ok=True)

    # Resize images
    image_files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(('.png','.jpg','.jpeg')))
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        output_img_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_img_path, resized)

    print(f"[Resize] Video{video_id}: {len(image_files)} images resized → {output_folder}")
    if labeled:
        print(f"[Resize] Labels copied → {config['output_labels']}")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    resize_images(1)  # Video1
    resize_images(2)  # Video2