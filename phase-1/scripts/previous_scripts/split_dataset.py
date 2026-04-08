import os
import random
import shutil

# Paths
BASE_DIR = r'C:\Users\ekta\MTech\Capstone\crane-ai\phase-1\data\clean_data\video1_clean_training'
# The subfolders YOLOv8 looks for
FOLDERS = ['train/images', 'train/labels', 'val/images', 'val/labels']

def split_data(train_ratio=0.8):
    for folder in FOLDERS:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

    # Get all clean images
    images = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(('.jpg', '.png'))]
    random.seed(42) # For reproducible results
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    def move_files(files, section):
        for filename in files:
            # Move Image
            shutil.move(os.path.join(BASE_DIR, filename), 
                        os.path.join(BASE_DIR, section, 'images', filename))
            # Move corresponding Label
            label_name = os.path.splitext(filename)[0] + '.txt'
            src_label = os.path.join(BASE_DIR, label_name)
            if os.path.exists(src_label):
                shutil.move(src_label, os.path.join(BASE_DIR, section, 'labels', label_name))

    move_files(train_files, 'train')
    move_files(val_files, 'val')
    print(f"Split Complete: {len(train_files)} Training, {len(val_files)} Validation.")

if __name__ == "__main__":
    split_data()