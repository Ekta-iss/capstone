# ------------------------------
# Merge CVAT and Auto-labeled Video1 data
# ------------------------------

import os
import shutil
import hashlib

# ------------------------------
# Paths
# ------------------------------
CVAT_IMAGES = "../../data/cvat/video1/obj_train_data/images"
CVAT_LABELS = "../../data/cvat/video1/obj_train_data/labels"

AUTO_IMAGES = "../../data/auto_labels/video1/images"
AUTO_LABELS = "../../data/auto_labels/video1/labels"

MERGED_IMAGES = "../../data/combined/video1/images"
MERGED_LABELS = "../../data/combined/video1/labels"

os.makedirs(MERGED_IMAGES, exist_ok=True)
os.makedirs(MERGED_LABELS, exist_ok=True)

# ------------------------------
# Helper functions
# ------------------------------

def hash_image(path):
    """Return MD5 hash of an image"""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def copy_with_hash_check(src_img_folder, src_label_folder):
    """Copy images and labels to merged folder with duplicate handling"""
    existing_hashes = set()
    existing_filenames = set(os.listdir(MERGED_IMAGES))

    for fname in sorted(os.listdir(src_img_folder)):
        src_img_path = os.path.join(src_img_folder, fname)
        src_label_path = os.path.join(src_label_folder, fname.replace(".jpg", ".txt"))

        if not os.path.isfile(src_label_path):
            # Skip images without labels
            continue

        # Compute hash
        img_hash = hash_image(src_img_path)
        if img_hash in existing_hashes:
            continue  # skip truly identical images
        existing_hashes.add(img_hash)

        # Handle duplicate filenames
        base, ext = os.path.splitext(fname)
        new_fname = fname
        counter = 1
        while new_fname in existing_filenames:
            new_fname = f"{base}_{counter}{ext}"
            counter += 1
        existing_filenames.add(new_fname)

        # Copy image
        shutil.copy2(src_img_path, os.path.join(MERGED_IMAGES, new_fname))
        # Copy label
        new_label_fname = new_fname.replace(".jpg", ".txt")
        shutil.copy2(src_label_path, os.path.join(MERGED_LABELS, new_label_fname))

# ------------------------------
# Copy CVAT data
# ------------------------------
copy_with_hash_check(CVAT_IMAGES, CVAT_LABELS)

# ------------------------------
# Copy Auto-labeled data
# ------------------------------
copy_with_hash_check(AUTO_IMAGES, AUTO_LABELS)

print("✅ Merge completed!")
print(f"Total images in merged dataset: {len(os.listdir(MERGED_IMAGES))}")
print(f"Total labels in merged dataset: {len(os.listdir(MERGED_LABELS))}")