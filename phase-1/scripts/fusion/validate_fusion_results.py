import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# =========================
# PATH
# =========================
FUSION_PATH = "../../data/fusion/fusion_results.npy"
OUTPUT_DIR = "../../data/evaluation/fusion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
def load_data():
    data = np.load(FUSION_PATH, allow_pickle=True)
    return pd.DataFrame(data.tolist())

# =========================
# SAFETY LABELS
# =========================
def map_safety(risk):
    if risk > 0.7:
        return "CRITICAL"
    elif risk > 0.4:
        return "WARNING"
    return "SAFE"

# =========================
# ANALYSIS
# =========================
def analyze(df):

    df["safety"] = df["risk_score"].apply(map_safety)

    # Save CSV summary
    df.to_csv(os.path.join(OUTPUT_DIR, "fusion_summary.csv"), index=False)

    print("📊 Generating plots...")

    # =========================
    # 1. Risk Distribution
    # =========================
    plt.figure()
    plt.hist(df["risk_score"], bins=20)
    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, "risk_distribution.png"))

    # =========================
    # 2. Safety Level Pie Chart
    # =========================
    plt.figure()
    df["safety"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Safety Level Distribution")
    plt.ylabel("")
    plt.savefig(os.path.join(OUTPUT_DIR, "safety_pie.png"))

    # =========================
    # 3. Class-wise Risk
    # =========================
    plt.figure()
    df.groupby("class")["risk_score"].mean().plot(kind="bar")
    plt.title("Average Risk per Class")
    plt.xlabel("Class")
    plt.ylabel("Risk Score")
    plt.savefig(os.path.join(OUTPUT_DIR, "class_risk.png"))

    # =========================
    # 4. Confidence vs Risk
    # =========================
    plt.figure()
    plt.scatter(df["confidence"], df["risk_score"])
    plt.title("Confidence vs Risk Score")
    plt.xlabel("Confidence")
    plt.ylabel("Risk Score")
    plt.savefig(os.path.join(OUTPUT_DIR, "conf_vs_risk.png"))

    # =========================
    # 5. Risk over Frames
    # =========================
    plt.figure()
    df["frame_idx"] = range(len(df))
    plt.plot(df["frame_idx"], df["risk_score"])
    plt.title("Risk Over Time (Frames)")
    plt.xlabel("Frame")
    plt.ylabel("Risk Score")
    plt.savefig(os.path.join(OUTPUT_DIR, "risk_timeline.png"))

    print(f"✅ Saved all outputs to: {OUTPUT_DIR}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    df = load_data()
    analyze(df)