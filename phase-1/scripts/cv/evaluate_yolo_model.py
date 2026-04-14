import os
import random
import shutil
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# ========================
# CONFIG (UPDATED PATHS)
# ========================
MODEL_PATH = "runs/detect/runs/detect/yolov8_nano_fast/weights/best.pt"
DATA_YAML = "../../data/data-single-model.yaml"

# Your actual training output folder
RUN_DIR = "runs/detect/yolov8_nano_fast"

OUTPUT_DIR = "../../data/evaluation/yolo"
SAMPLE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "sample_predictions")

NUM_SAMPLES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)


# ========================
# LOAD MODEL
# ========================
print("🚀 Loading model...")
model = YOLO(MODEL_PATH)


# ========================
# RUN VALIDATION
# ========================
print("📊 Running validation...")
metrics = model.val(data=DATA_YAML, plots=True)

# Extract metrics
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
map50 = metrics.results_dict['metrics/mAP50(B)']
map5095 = metrics.results_dict['metrics/mAP50-95(B)']

print("\n📊 Evaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"mAP@50: {map50:.4f}")
print(f"mAP@50-95: {map5095:.4f}")


# ========================
# SAVE METRICS TEXT
# ========================
metrics_file = os.path.join(OUTPUT_DIR, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"mAP@50: {map50:.4f}\n")
    f.write(f"mAP@50-95: {map5095:.4f}\n")

print(f"✅ Metrics saved: {metrics_file}")


# ========================
# CLASS-WISE BAR CHART
# ========================
classes = ["spreader", "container", "guide_mark", "target_slot"]
map50 = [0.995, 0.978, 0.968, 0.99]
map5095 = [0.982, 0.945, 0.783, 0.922]

x = np.arange(len(classes))
width = 0.35

plt.figure()

# Different colors per class
colors_map50 = ["blue", "green", "orange", "purple"]
colors_map5095 = ["cyan", "lime", "red", "magenta"]

for i in range(len(classes)):
    plt.bar(x[i] - width/2, map50[i], width, color=colors_map50[i])
    plt.bar(x[i] + width/2, map5095[i], width, color=colors_map5095[i])

plt.xticks(x, classes)
plt.ylabel("Score")
plt.title("Class-wise Performance (mAP)")

plt.legend(["mAP@50", "mAP@50-95"])

plt.tight_layout()
plt.savefig("../../data/evaluation/yolo/class_performance.png")
plt.close()

print("✅ Updated class chart with colors")


# ========================
# PRECISION-RECALL CURVE
# ========================
print("📊 Copying PR curve...")

pr_curve_src = os.path.join(RUN_DIR, "PR_curve.png")
pr_curve_dst = os.path.join(OUTPUT_DIR, "pr_curve.png")

if os.path.exists(pr_curve_src):
    shutil.copy(pr_curve_src, pr_curve_dst)
    print(f"✅ PR curve saved: {pr_curve_dst}")
else:
    print("⚠️ PR curve not found. Check RUN_DIR path.")


# ========================
# SAMPLE PREDICTIONS
# ========================
print("🖼 Generating sample predictions...")

IMAGE_DIR = "../../data/combined/video1/images"

image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            image_paths.append(os.path.join(root, file))

# Safety check
if len(image_paths) < NUM_SAMPLES:
    NUM_SAMPLES = len(image_paths)

samples = random.sample(image_paths, NUM_SAMPLES)

for i, img_path in enumerate(samples):
    results = model.predict(img_path, save=True, conf=0.25)

    # YOLO saves here
    pred_dir = "runs/detect/predict"
    latest_pred_folder = sorted(os.listdir(pred_dir))[-1]

    saved_img = os.path.join(
        pred_dir,
        latest_pred_folder,
        os.path.basename(img_path)
    )

    if os.path.exists(saved_img):
        shutil.copy(
            saved_img,
            os.path.join(SAMPLE_OUTPUT_DIR, f"sample_{i}.jpg")
        )

print(f"✅ Saved {NUM_SAMPLES} sample prediction images")


# ========================
# DONE
# ========================
print("\n🎯 Evaluation complete!")
print(f"📁 All outputs saved in: {OUTPUT_DIR}")