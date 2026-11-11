
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_PATH = PROJECT_DIR / "dataset" / "y.npy"
GRAPH_DIR = PROJECT_DIR / "graphs"

GRAPH_DIR.mkdir(exist_ok=True)

# Load emotion labels
y = np.load(DATASET_PATH, allow_pickle=True)

# Count samples per emotion
emotion_counts = Counter(y)

emotions = list(emotion_counts.keys())
counts = list(emotion_counts.values())

# Plot graph
plt.figure(figsize=(10, 6))

bars = plt.bar(emotions, counts)

plt.title("Emotion Class Distribution in RAVDESS Dataset")
plt.xlabel("Emotion Categories")
plt.ylabel("Number of Audio Samples")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Display values above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(height),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save graph
output_path = GRAPH_DIR / "emotion_class_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print("Emotion Distribution:")
for name, count in emotion_counts.items():
    print(f"{name}: {count}")

print(f"\nGraph saved at: {output_path}")
