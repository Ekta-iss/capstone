# scripts/fusion/train_fusion_lstm.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================
DATA_X = "../../data/fusion/lstm_X.npy"
DATA_Y = "../../data/fusion/lstm_Y.npy"

MODEL_DIR = "../../data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "fusion_lstm.pth")

PRED_PATH = os.path.join(MODEL_DIR, "lstm_predictions.npy")
GT_PATH = os.path.join(MODEL_DIR, "lstm_ground_truth.npy")
LOSS_PLOT = os.path.join(MODEL_DIR, "lstm_loss_curve.png")

BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# MODEL
# =========================
class FusionLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # future_distance, angle, risk
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# =========================
# LOAD DATA
# =========================
def load_data():
    X = np.load(DATA_X)
    Y = np.load(DATA_Y)
    return X, Y


# =========================
# TRAIN FUNCTION
# =========================
def train(model, train_loader, val_loader):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.to(DEVICE)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(DEVICE), Y.to(DEVICE)
                pred = model(X)
                loss = criterion(pred, Y)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    return train_losses, val_losses


# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred):

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print("\n📊 FINAL METRICS")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")


# =========================
# PLOT LOSS
# =========================
def plot_loss(train_loss, val_loss):

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("LSTM Fusion Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    plt.savefig(LOSS_PLOT)
    print(f"📊 Loss curve saved: {LOSS_PLOT}")


# =========================
# MAIN
# =========================
def main():

    print("🔗 Loading LSTM Fusion Dataset...")

    X, Y = load_data()

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # =========================
    # TRAIN / VAL SPLIT
    # =========================
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=BATCH_SIZE
    )

    # =========================
    # MODEL
    # =========================
    model = FusionLSTM()

    print("🚀 Training Fusion LSTM Model...")
    train_loss, val_loss = train(model, train_loader, val_loader)

    # =========================
    # SAVE MODEL
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"✅ Model saved at: {MODEL_PATH}")

    # =========================
    # SAMPLE PREDICTIONS
    # =========================
    model.eval()

    with torch.no_grad():
        sample_X = X_val[:200].to(DEVICE)
        preds = model(sample_X).cpu().numpy()
        gt = Y_val[:200].numpy()

    np.save(PRED_PATH, preds)
    np.save(GT_PATH, gt)

    print(f"📦 Predictions saved: {PRED_PATH}")
    print(f"📦 Ground truth saved: {GT_PATH}")

    # =========================
    # METRICS
    # =========================
    compute_metrics(gt, preds)

    # =========================
    # VISUALIZATION
    # =========================
    plot_loss(train_loss, val_loss)

    print("\n🎯 TRAINING COMPLETE!")


if __name__ == "__main__":
    main()