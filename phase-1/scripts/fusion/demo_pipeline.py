import os
import json
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = "../cv/runs/detect/runs/detect/yolov8_nano_fast/weights/best.pt"
IMAGE_FOLDER = "../../data/combined/video1/images"

DEBUG = True
FRAME_TIME = 0.1  # seconds per frame


# =========================
# YOLO INFERENCE
# =========================
def run_yolo(yolo, img_path):
    results = yolo.predict(img_path, verbose=False)[0]

    detections = []
    for b in results.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append({
            "class": cls,
            "conf": conf,
            "cx": cx,
            "cy": cy
        })

    return detections


# =========================
# CRANE CYCLE TRACKER
# =========================
class CraneCycleTracker:
    def __init__(self):
        self.in_cycle = False
        self.cycles = 0

        self.idle_time = 0.0
        self.safety_events = 0

        self.last_active_frame = 0
        self.cycle_start = None
        self.cycle_durations = []

        # smoothing
        self.activity_window = []
        self.window_size = 5

        # 🔥 inactivity tolerance (frames)
        self.inactive_threshold = 10

        self.debug_log = []

    # =========================
    # ACTIVITY LOGIC (FIXED)
    # =========================
    def is_active(self, detections):

        confident = [d for d in detections if d["conf"] > 0.5]

        # 🔥 IMPORTANT: require meaningful detections
        active = len(confident) >= 2

        self.activity_window.append(active)

        if len(self.activity_window) > self.window_size:
            self.activity_window.pop(0)

        smoothed = sum(self.activity_window) >= 3

        return smoothed, len(confident)

    # =========================
    # UPDATE PER FRAME
    # =========================
    def update(self, detections, frame_idx):

        smoothed_active, conf_count = self.is_active(detections)

        if DEBUG:
            print(f"\n📦 Frame {frame_idx}")
            print(f"Detections: {len(detections)} | Confident: {conf_count} | Active: {smoothed_active}")
            print(f"In cycle: {self.in_cycle}")

        # =========================
        # START CYCLE
        # =========================
        if smoothed_active and not self.in_cycle:
            self.in_cycle = True
            self.cycles += 1
            self.cycle_start = frame_idx

            idle_gap = (frame_idx - self.last_active_frame) * FRAME_TIME
            self.idle_time += max(0, idle_gap)

            if DEBUG:
                print(f"🚀 CYCLE START #{self.cycles}")
                print(f"⏱ Idle added: {idle_gap:.2f}s")

        # =========================
        # SAFETY EVENTS
        # =========================
        low_conf = sum(1 for d in detections if d["conf"] < 0.25)

        if low_conf > 3:
            self.safety_events += 1
            if DEBUG:
                print("⚠️ SAFETY EVENT")

        # =========================
        # END CYCLE (FIXED)
        # =========================
        if not smoothed_active and self.in_cycle:

            inactive_gap = frame_idx - self.last_active_frame

            if inactive_gap > self.inactive_threshold:
                self.in_cycle = False
                self.last_active_frame = frame_idx

                if self.cycle_start is not None:
                    duration = frame_idx - self.cycle_start
                    self.cycle_durations.append(duration)

                if DEBUG:
                    print("🏁 CYCLE END")

        # update last active frame
        if smoothed_active:
            self.last_active_frame = frame_idx

        # =========================
        # DEBUG LOG
        # =========================
        self.debug_log.append({
            "frame": frame_idx,
            "detections": len(detections),
            "confident": conf_count,
            "active": smoothed_active,
            "in_cycle": self.in_cycle,
            "cycles": self.cycles,
            "idle_time": self.idle_time,
            "safety_events": self.safety_events
        })


# =========================
# MAIN PIPELINE
# =========================
def main():

    print("🔗 Loading YOLO model...")
    yolo = YOLO(YOLO_MODEL_PATH)

    images = sorted(os.listdir(IMAGE_FOLDER))
    tracker = CraneCycleTracker()

    print("🚀 Running DEMO PIPELINE...\n")

    for i, img in enumerate(images):
        img_path = os.path.join(IMAGE_FOLDER, img)

        detections = run_yolo(yolo, img_path)
        tracker.update(detections, i)

    # =========================
    # CLOSE LAST CYCLE (FIX)
    # =========================
    if tracker.in_cycle and tracker.cycle_start is not None:
        duration = len(images) - tracker.cycle_start
        tracker.cycle_durations.append(duration)

    # =========================
    # FINAL KPI REPORT
    # =========================
    avg_frames = float(np.mean(tracker.cycle_durations)) if tracker.cycle_durations else 0

    kpi = {
        "idle_time_sec": round(tracker.idle_time, 2),
        "throughput_cycles": tracker.cycles,
        "safety_events": tracker.safety_events,
        "avg_cycle_time_frames": avg_frames,
        "avg_cycle_time_sec": round(avg_frames * FRAME_TIME, 2)
    }

    print("\n📊 FINAL KPI REPORT:")
    print(json.dumps(kpi, indent=4))

    # =========================
    # SAVE OUTPUT
    # =========================
    out_dir = "../../data/fusion"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "demo_kpi.json"), "w") as f:
        json.dump(kpi, f, indent=4)

    with open(os.path.join(out_dir, "demo_debug_log.json"), "w") as f:
        json.dump(tracker.debug_log, f, indent=4)

    print(f"\n📁 Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()