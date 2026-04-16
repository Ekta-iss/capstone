import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os

# ========================
# 1. CONFIG & PATHS
# ========================
DATA_DIR = "../fusion/"  
MODEL_SAVE_PATH = "../../models/fusion_dashboard_mlp.pth"
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001

# SCALE FACTOR: We divide ETA by 50 to bring it into the 0-1 range
# (Assuming your max cycle time is around 50-100 seconds)
ETA_SCALE = 50.0 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 2. MODEL DEFINITION
# ========================
class DashboardMLP(nn.Module):
    def __init__(self, input_size=6):
        super(DashboardMLP, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # ETA Head (Regression)
        self.eta_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Mode Head (Classification: 0=Auto, 1=Assisted, 2=Manual)
        self.mode_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
        )

    def forward(self, x):
        features = self.shared(x)
        eta = self.eta_head(features)
        mode_logits = self.mode_head(features)
        return eta, mode_logits

# ========================
# 3. TRAINING WITH LOSS BALANCING
# ========================
def train():
    print(f"🖥️ Training on: {DEVICE}")

    # Load data
    x_path = os.path.join(DATA_DIR, "fusion_X_final_auto.npy")
    y_path = os.path.join(DATA_DIR, "fusion_Y_final_auto.npy")
    
    X = np.load(x_path)
    Y = np.load(y_path)

    # --- SCALING ---
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # Scale ETA by our constant to reduce loss magnitude
    y_eta = torch.tensor(Y[:, 0] / ETA_SCALE, dtype=torch.float32).unsqueeze(1)
    y_mode = torch.tensor(Y[:, 1], dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_eta, y_mode)
    
    # Split into Train/Val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = DashboardMLP(input_size=6).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_eta = nn.MSELoss()
    criterion_mode = nn.CrossEntropyLoss()

    print(f"🚀 Training Multi-Task MLP on {len(train_ds)} samples...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_x, batch_eta, batch_mode in train_loader:
            batch_x, batch_eta, batch_mode = batch_x.to(DEVICE), batch_eta.to(DEVICE), batch_mode.to(DEVICE)
            
            optimizer.zero_grad()
            pred_eta, pred_mode_logits = model(batch_x)
            
            # --- WEIGHTED LOSS ---
            # We weight the Mode higher because safety is more important than ETA precision
            loss_eta = criterion_eta(pred_eta, batch_eta)
            loss_mode = criterion_mode(pred_mode_logits, batch_mode)
            
            loss = (1.0 * loss_eta) + (2.0 * loss_mode) 
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, be, bm in val_loader:
                    bx, be, bm = bx.to(DEVICE), be.to(DEVICE), bm.to(DEVICE)
                    pe, pm = model(bx)
                    val_loss += (criterion_eta(pe, be) + criterion_mode(pm, bm)).item()
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    # Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Dashboard MLP saved with Scaling Factor: {ETA_SCALE}")

if __name__ == "__main__":
    train()