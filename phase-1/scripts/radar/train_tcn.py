# scripts/radar/train_tcn.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ========================
# CONFIG
# ========================
DATA_DIR = "../../data/radar/processed"

SEQ_LEN = 40
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# LOAD DATA
# ========================
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    return X_train, y_train, X_val, y_val


# ========================
# TCN BLOCK (RESIDUAL)
# ========================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.relu(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ========================
# TCN MODEL
# ========================
class TCN(nn.Module):
    def __init__(self, input_size=3, hidden=128):
        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(input_size, hidden, dilation=1),
            TCNBlock(hidden, hidden, dilation=2),
            TCNBlock(hidden, hidden, dilation=4),
            TCNBlock(hidden, hidden, dilation=8),
        )

        self.fc = nn.Linear(hidden, 3)

    def forward(self, x):
        # x: (batch, seq, features)
        x = x.permute(0, 2, 1)  # → (batch, features, seq)
        x = self.tcn(x)
        x = x[:, :, -1]         # last timestep
        return self.fc(x)


# ========================
# CUSTOM LOSS (OPTIONAL WEIGHTING)
# ========================
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, weights=None):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction='none')
        self.weights = weights if weights is not None else torch.tensor([1.0, 1.5, 1.5])

    def forward(self, pred, target):
        loss = self.huber(pred, target)
        return (loss * self.weights.to(loss.device)).mean()


# ========================
# TRAIN FUNCTION
# ========================
def train_model(model, train_loader, val_loader):

    # Try weighted loss (helps velocity)
    criterion = WeightedHuberLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Val Loss: {val_loss/len(val_loader):.4f}")


# ========================
# MAIN
# ========================
def main():

    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data()

    print("Shapes:")
    print("X_train:", X_train.shape)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # DataLoader
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = TCN(input_size=3, hidden=128)

    print("Training improved TCN model...")
    train_model(model, train_loader, val_loader)

    # Save model
    os.makedirs("../../models", exist_ok=True)
    torch.save(model.state_dict(), "../../models/tcn_radar_improved.pth")

    print("✅ Model saved to ../../models/tcn_radar_improved.pth")


if __name__ == "__main__":
    main()