from ultralytics import YOLO
import os
import shutil

# --- 1. Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Target Folders
DATA_YAML_PATH = os.path.join(SCRIPT_DIR, 'data.yaml')
RUNS_DIR = os.path.join(PHASE1_DIR, 'runs')
MODELS_DIR = os.path.join(PHASE1_DIR, 'models')

def train_on_cpu():
    print(f"--- Starting Port-AIS Model Training (CPU Mode) ---")
    
    # Ensure the models folder exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load the Nano Model (Lightweight for CPU)
    model = YOLO('yolov8n.pt')

    # --- 2. Start Training ---
    # project: The root directory for all training runs
    # name: The specific subfolder for this experiment
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=50,
        imgsz=640,
        batch=4,
        workers=0,
        device='cpu',
        plots=True,
        project=RUNS_DIR, 
        name='scass_v1_experiment'
    )

    # --- 3. Save a Copy to the 'models' Folder ---
    # YOLOv8 saves the best weights in: project/name/weights/best.pt
    trained_weights_path = os.path.join(RUNS_DIR, 'scass_v1_experiment', 'weights', 'best.pt')
    final_model_path = os.path.join(MODELS_DIR, 'scass_best_v1.pt')

    if os.path.exists(trained_weights_path):
        shutil.copy(trained_weights_path, final_model_path)
        print(f"\nSuccess! Final model saved to: {final_model_path}")
    else:
        print("\nWarning: Could not find trained weights to copy.")

    print(f"Full training logs available in: {os.path.join(RUNS_DIR, 'scass_v1_experiment')}")

if __name__ == "__main__":
    train_on_cpu()