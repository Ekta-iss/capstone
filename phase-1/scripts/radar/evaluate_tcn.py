# scripts/radar/evaluate_tcn.py

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========================
# CONFIG
# ========================
DATA_DIR = "../../data/radar/processed"
MODEL_PATH = "../../models/tcn_radar.pth"
OUTPUT_DIR = "../../outputs/radar_eval"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),
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
def compute_feature_metrics(y_true, y_pred, idx):
    mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
    rmse = np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))
    r2 = r2_score(y_true[:, idx], y_pred[:, idx])
    return mae, rmse, r2


def compute_metrics(y_true, y_pred):
    print("\n📊 MODEL PERFORMANCE")

    overall_mse = np.mean((y_true - y_pred) ** 2)
    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_rmse = np.sqrt(overall_mse)

    print(f"Overall MSE  : {overall_mse:.6f}")
    print(f"Overall MAE  : {overall_mae:.6f}")
    print(f"Overall RMSE : {overall_rmse:.6f}")

    labels = ["Distance", "Angle", "Velocity"]
    metrics_dict = {}

    for i, label in enumerate(labels):
        mae, rmse, r2 = compute_feature_metrics(y_true, y_pred, i)
        metrics_dict[label] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        print(f"\n{label} Metrics:")
        print(f"  MAE  : {mae:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        print(f"  R²   : {r2:.6f}")

    euclidean_error = np.linalg.norm(y_true - y_pred, axis=1)
    print(f"\nEuclidean Error Magnitude:")
    print(f"  Mean  : {np.mean(euclidean_error):.6f}")
    print(f"  Std   : {np.std(euclidean_error):.6f}")
    print(f"  Max   : {np.max(euclidean_error):.6f}")

    return metrics_dict, euclidean_error


# ========================
# PLOTS FOR PPT
# ========================
def add_metric_box(ax, mae, rmse, r2):
    textstr = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}"
    ax.text(
        0.02, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
    )


def plot_results(y_true, y_pred, metrics_dict, euclidean_error, num_samples=200):
    plt.style.use("default")

    # --- Distance Prediction ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(y_true[:num_samples, 0], label="Actual", linewidth=2)
    ax.plot(y_pred[:num_samples, 0], label="Predicted", linewidth=2)
    ax.set_title("Distance Prediction", fontsize=13)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    add_metric_box(
        ax,
        metrics_dict["Distance"]["MAE"],
        metrics_dict["Distance"]["RMSE"],
        metrics_dict["Distance"]["R2"]
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "distance_prediction.png"), dpi=300)
    plt.close(fig)

    # --- Velocity Prediction ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(y_true[:num_samples, 2], label="Actual", linewidth=1.8)
    ax.plot(y_pred[:num_samples, 2], label="Predicted", linewidth=1.8)
    ax.set_title("Velocity Prediction", fontsize=13)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    add_metric_box(
        ax,
        metrics_dict["Velocity"]["MAE"],
        metrics_dict["Velocity"]["RMSE"],
        metrics_dict["Velocity"]["R2"]
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "velocity_prediction.png"), dpi=300)
    plt.close(fig)

    # --- Error Distribution ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(euclidean_error, bins=30, alpha=0.85, edgecolor="black")
    ax.set_title("Euclidean Error Magnitude Distribution", fontsize=13)
    ax.set_xlabel("Euclidean Error Magnitude")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

    mean_err = np.mean(euclidean_error)
    std_err = np.std(euclidean_error)
    ax.text(
        0.02, 0.98,
        f"Mean: {mean_err:.3f}\nStd: {std_err:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"), dpi=300)
    plt.close(fig)


# ========================
# MAIN
# ========================
def main():
    print("Loading test data...")
    X_test, y_test = load_data()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    print("Loading model...")
    model = TCN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()

    print("Running predictions...")
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    print("Evaluating performance...")
    metrics_dict, euclidean_error = compute_metrics(y_test, y_pred)

    print("\nGenerating plots for PPT...")
    plot_results(y_test, y_pred, metrics_dict, euclidean_error)

    print("\n✅ Evaluation complete!")
    print("Saved plots:")
    print(f"- {os.path.join(OUTPUT_DIR, 'distance_prediction.png')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'velocity_prediction.png')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'error_distribution.png')}")


if __name__ == "__main__":
    main()