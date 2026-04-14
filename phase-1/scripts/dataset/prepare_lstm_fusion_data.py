# scripts/dataset/prepare_lstm_fusion_data.py

import os
import numpy as np

# =========================
# CONFIG
# =========================
MLP_X_PATH = "../../data/fusion/mlp_X.npy"
MLP_Y_PATH = "../../data/fusion/mlp_Y.npy"

OUTPUT_X = "../../data/fusion/lstm_X.npy"
OUTPUT_Y = "../../data/fusion/lstm_Y.npy"

SEQ_LEN = 20   # can tune: 10 / 20 / 40


# =========================
# LOAD DATA
# =========================
def load_data():
    X = np.load(MLP_X_PATH)
    Y = np.load(MLP_Y_PATH)
    return X, Y


# =========================
# CREATE SEQUENCES
# =========================
def create_sequences(X, Y, seq_len=20):

    X_seq = []
    Y_seq = []

    print("🔗 Creating temporal sequences...")

    for i in range(len(X) - seq_len):

        x_window = X[i:i+seq_len]
        y_target = Y[i+seq_len]   # predict future step

        X_seq.append(x_window)
        Y_seq.append(y_target)

    return np.array(X_seq, dtype=np.float32), np.array(Y_seq, dtype=np.float32)


# =========================
# MAIN
# =========================
def main():

    print("🔗 Preparing LSTM Fusion Dataset...")

    X, Y = load_data()

    print(f"Original X shape: {X.shape}")
    print(f"Original Y shape: {Y.shape}")

    X_lstm, Y_lstm = create_sequences(X, Y, SEQ_LEN)

    os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)

    np.save(OUTPUT_X, X_lstm)
    np.save(OUTPUT_Y, Y_lstm)

    print("✅ LSTM dataset created successfully")
    print("X_lstm shape:", X_lstm.shape)
    print("Y_lstm shape:", Y_lstm.shape)

    print("\n📌 Example:")
    print("Each sample = sequence of", SEQ_LEN, "frames")
    print("Each feature vector = [cls, conf, dist, angle, velocity, cx, cy]")


if __name__ == "__main__":
    main()