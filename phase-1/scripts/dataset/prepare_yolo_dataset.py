# scripts/dataset/prepare_yolo_dataset.py

import os
import random
from sklearn.model_selection import train_test_split

# ========================
# CONFIG
# ========================
IMAGE_DIR = "../../data/combined/video1/images"
LABEL_DIR = "../../data/combined/video1/labels"

OUTPUT_DIR = "../../data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

VALID_EXT = (".jpg", ".jpeg", ".png")


# ========================
# STEP 1: COLLECT VALID PAIRS
# ========================
def collect_valid_pairs():
    valid_images = []

    print("🔍 Scanning dataset...")

    for file in os.listdir(IMAGE_DIR):
        if not file.lower().endswith(VALID_EXT):
            continue

        img_path = os.path.join(IMAGE_DIR, file)
        label_path = os.path.join(LABEL_DIR, file.replace(".jpg", ".txt"))

        # Check label exists
        if not os.path.exists(label_path):
            continue

        # Check label not empty
        if os.path.getsize(label_path) == 0:
            continue

        valid_images.append(img_path)

    print(f"✅ Valid image-label pairs: {len(valid_images)}")
    return valid_images


# ========================
# STEP 2: SPLIT DATA
# ========================
def split_data(images):

    train, temp = train_test_split(
        images,
        test_size=(1 - TRAIN_RATIO),
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42
    )

    return train, val, test


# ========================
# STEP 3: SAVE SPLITS
# ========================
def save_split(file_list, path):
    with open(path, "w") as f:
        for item in file_list:
            f.write(item + "\n")


# ========================
# MAIN
# ========================
def main():

    images = collect_valid_pairs()

    if len(images) == 0:
        print("❌ No valid data found!")
        return

    train, val, test = split_data(images)

    print(f"\n📊 Split:")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")

    save_split(train, os.path.join(OUTPUT_DIR, "train.txt"))
    save_split(val, os.path.join(OUTPUT_DIR, "val.txt"))
    save_split(test, os.path.join(OUTPUT_DIR, "test.txt"))

    print("\n✅ Split files created in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()