# scripts/cv/train_yolo_model.py

from ultralytics import YOLO
import shutil
import os

# ========================
# CONFIG
# ========================
MODEL_NAME = "yolov8n.pt"
DATA_YAML = "../../data/data-single-model.yaml"

EPOCHS = 20
IMG_SIZE = 416
BATCH_SIZE = 8
DEVICE = "cpu"

# Output folders
PROJECT_DIR = "runs/detect"
RUN_NAME = "yolov8_nano_fast"

# Final model save location
MODEL_SAVE_DIR = "../../data/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ========================
# MAIN
# ========================
def main():

    print("🚀 YOLOv8 Nano Training (CPU Optimized)")

    model = YOLO(MODEL_NAME)

    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        # Performance
        workers=0,
        cache=False,
        amp=False,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,

        # Early stopping
        patience=5,

        # Logging
        project=PROJECT_DIR,
        name=RUN_NAME,
        verbose=True,
        plots=True
    )

    print("\n✅ Training Completed!")

    # ========================
    # COPY MODEL TO data/models
    # ========================
    run_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights")

    best_model = os.path.join(run_path, "best.pt")
    last_model = os.path.join(run_path, "last.pt")

    if os.path.exists(best_model):
        shutil.copy(best_model, os.path.join(MODEL_SAVE_DIR, "yolov8_best.pt"))
        print(f"✅ Best model saved to: {MODEL_SAVE_DIR}/yolov8_best.pt")

    if os.path.exists(last_model):
        shutil.copy(last_model, os.path.join(MODEL_SAVE_DIR, "yolov8_last.pt"))
        print(f"✅ Last model saved to: {MODEL_SAVE_DIR}/yolov8_last.pt")

    # ========================
    # VALIDATION
    # ========================
    print("\n📊 Running Validation...")
    model.val()


if __name__ == "__main__":
    main()