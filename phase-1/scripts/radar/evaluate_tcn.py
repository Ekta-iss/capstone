# scripts/radar/evaluate_tcn.py

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ========================
# CONFIG
# ========================
DATA_DIR = "../../data/radar/processed"
MODEL_PATH = "../../models/tcn_radar.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# LOAD DATA
# ========================
def load_data():
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_test, y_test


# ========================
# SAME MODEL ARCHITECTURE
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

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class TCN(nn.Module):
    def __init__(self, input_size=3, hidden=64):
        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(input_size, hidden, dilation=1),
            TCNBlock(hidden, hidden, dilation=2),
            TCNBlock(hidden, hidden, dilation=4),
        )

        self.fc = nn.Linear(hidden, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]
        return self.fc(x)


# ========================
# METRICS
# ========================
def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    print("\n📊 MODEL PERFORMANCE")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    # per feature
    labels = ["Distance", "Angle", "Velocity"]

    for i in range(3):
        err = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        print(f"{labels[i]} MAE: {err:.6f}")


# ========================
# PLOTS FOR PPT
# ========================
def plot_results(y_true, y_pred):

    # --- Distance ---
    plt.figure()
    plt.plot(y_true[:200, 0], label="Actual")
    plt.plot(y_pred[:200, 0], label="Predicted")
    plt.title("Distance Prediction")
    plt.legend()
    plt.savefig("distance_prediction.png")
    plt.show()

    # --- Velocity ---
    plt.figure()
    plt.plot(y_true[:200, 2], label="Actual")
    plt.plot(y_pred[:200, 2], label="Predicted")
    plt.title("Velocity Prediction")
    plt.legend()
    plt.savefig("velocity_prediction.png")
    plt.show()

    # --- Error distribution ---
    error = np.linalg.norm(y_true - y_pred, axis=1)

    plt.figure()
    plt.hist(error, bins=30)
    plt.title("Prediction Error Distribution")
    plt.savefig("error_distribution.png")
    plt.show()


# ========================
# MAIN
# ========================
def main():

    print("Loading test data...")
    X_test, y_test = load_data()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    print("Loading model...")
    model = TCN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE),
    strict=False)
    model.eval()

    print("Running predictions...")
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    print("Evaluating performance...")
    compute_metrics(y_test, y_pred)

    print("\nGenerating plots for PPT...")
    plot_results(y_test, y_pred)

    print("\n✅ Evaluation complete!")
    print("Saved plots:")
    print("- distance_prediction.png")
    print("- velocity_prediction.png")
    print("- error_distribution.png")


if __name__ == "__main__":
    main()