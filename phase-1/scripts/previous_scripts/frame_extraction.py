# Interval is set as 15 that is equal to 2FPS for a 30FPS video

import cv2
import os

def extract_frames(video_path, output_folder, interval=15):
    abs_video_path = os.path.abspath(video_path)
    abs_output_folder = os.path.abspath(output_folder)

    print(f"\nProcessing video: {abs_video_path}")
    print(f"Saving frames to: {abs_output_folder}")

    if not os.path.exists(video_path):
        print(f"ERROR: Video file does not exist: {abs_video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {abs_video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames reported by OpenCV: {total_frames}")

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        if frame is None:
            print(f"WARNING: Frame {count} is None, skipping.")
            count += 1
            continue

        if count % interval == 0:
            frame_filename = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)

            success = cv2.imwrite(frame_path, frame)
            if success:
                saved_count += 1
            else:
                print(f"ERROR: Failed to save frame to {os.path.abspath(frame_path)}")

        count += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames from {abs_video_path}")


# extract_frames("../data/videos/video_01.mp4", "../data/frames/video1_frames", interval=15)
extract_frames("../data/videos/video_02.mp4", "../data/frames/video2_frames", interval=15)