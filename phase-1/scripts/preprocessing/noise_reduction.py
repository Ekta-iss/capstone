#!/usr/bin/env python3

import cv2
import os
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

VIDEO_SETTINGS = {
    1: {
        "input_folder": os.path.join(BASE_DIR, "data/processed/video1/watermark_removed/images"),
        "output_folder": os.path.join(BASE_DIR, "data/processed/video1/denoised")
    },
    2: {
        "input_folder": os.path.join(BASE_DIR, "data/processed/video2/watermark_removed/images"),
        "output_folder": os.path.join(BASE_DIR, "data/processed/video2/denoised")
    }
}

# ------------------------------
# EDGE-AWARE DENOISING
# ------------------------------
def denoise_frame(frame):

    # Convert to float for precision
    img = frame.astype(np.float32) / 255.0

    # --- Step 1: Light bilateral (reduced smoothing)
    smooth = cv2.bilateralFilter(frame, d=5, sigmaColor=30, sigmaSpace=30)

    # --- Step 2: Edge map (detect important structures like lane lines)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Normalize edge mask
    edge_mask = edges.astype(np.float32) / 255.0
    edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 0)

    # Expand mask to 3 channels
    edge_mask = np.stack([edge_mask]*3, axis=-1)

    # --- Step 3: Blend (preserve edges, denoise flat areas)
    smooth = smooth.astype(np.float32) / 255.0
    blended = edge_mask * img + (1 - edge_mask) * smooth

    # --- Step 4: CLAHE only on L channel (very mild)
    blended_uint8 = (blended * 255).astype(np.uint8)
    lab = cv2.cvtColor(blended_uint8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- Step 5: Edge-aware sharpening (stronger on edges only)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.3, blur, -0.3, 0)

    # Blend sharpening only on edges
    sharp = sharp.astype(np.float32) / 255.0
    final = edge_mask * sharp + (1 - edge_mask) * blended

    final = (final * 255).astype(np.uint8)

    return final


# ------------------------------
# PROCESS FUNCTION
# ------------------------------
def process_denoise(video_id):
    input_folder = VIDEO_SETTINGS[video_id]["input_folder"]
    output_folder = VIDEO_SETTINGS[video_id]["output_folder"]

    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder does not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    for file in frame_files:
        frame_path = os.path.join(input_folder, file)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        denoised = denoise_frame(frame)

        # Keep SAME filename (CRITICAL for CVAT)
        cv2.imwrite(os.path.join(output_folder, file), denoised)

    print(f"[Noise Reduction] Processed {len(frame_files)} frames → {output_folder}")


if __name__ == "__main__":
    process_denoise(1)
    process_denoise(2)