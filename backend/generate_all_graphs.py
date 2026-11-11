
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

from pathlib import Path
from collections import Counter

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "dataset"
AUDIO_DIR = DATASET_DIR / "Audio_Speech_Actors_01-24"
GRAPH_DIR = PROJECT_DIR / "graphs"

GRAPH_DIR.mkdir(exist_ok=True)

# ---------------- FIGURE 1: CLASS DISTRIBUTION ----------------

y = np.load(DATASET_DIR / "y.npy", allow_pickle=True)

counts = Counter(y)
emotions = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(emotions, values)

plt.title("Emotion Class Distribution in RAVDESS Dataset")
plt.xlabel("Emotion Categories")
plt.ylabel("Number of Audio Samples")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(GRAPH_DIR / "paper_class_distribution.png", dpi=300)
plt.close()


# ---------------- FIGURE 2: WAVEFORM COMPARISON ----------------

emotion_codes = {
    "Neutral": "01",
    "Calm": "02",
    "Happy": "03",
    "Sad": "04",
    "Angry": "05",
    "Fearful": "06",
    "Disgust": "07",
    "Surprised": "08"
}

fig, axes = plt.subplots(4, 2, figsize=(15, 12))

for ax, (emotion, code) in zip(axes.flatten(), emotion_codes.items()):

    files = sorted(
        AUDIO_DIR.glob(f"Actor_*/*-*-{code}-*-*-*-*.wav")
    )

    if not files:
        ax.set_title(f"{emotion} - Audio not found")
        ax.axis("off")
        continue

    audio, sr = librosa.load(files[0], sr=None)

    time = np.arange(len(audio)) / sr

    ax.plot(time, audio, linewidth=0.5)
    ax.set_title(f"{emotion} Speech Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

plt.suptitle("Speech Waveform Comparison Across Eight Emotions")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "paper_waveform_comparison.png", dpi=300)
plt.close()



# ---------------- FIGURE 3: IMPROVED MFCC HEATMAP ----------------

sample_file = sorted(
    AUDIO_DIR.glob("Actor_01/03-01-03-01-01-01-01.wav")
)

if sample_file:
    audio, sr = librosa.load(sample_file[0], sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        mfcc,
        x_axis="time",
        y_axis="frames",
        cmap="coolwarm"
    )

    plt.colorbar(label="MFCC Coefficient Value")
    plt.title("MFCC Feature Representation of Speech Signal")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")

    plt.tight_layout()
    plt.savefig(
        GRAPH_DIR / "paper_mfcc_heatmap_v2.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Improved MFCC Heatmap saved.")



# ---------------- FIGURE 4: MODEL PERFORMANCE ----------------

csv_path = GRAPH_DIR / "model_comparison.csv"
df = pd.read_csv(csv_path)

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

for metric in metrics:
    df[metric] = pd.to_numeric(df[metric], errors="raise")

x = np.arange(len(df))
width = 0.18

plt.figure(figsize=(11, 6))

for i, metric in enumerate(metrics):
    plt.bar(
        x + (i - 1.5) * width,
        df[metric],
        width,
        label=metric
    )

plt.xticks(x, df["Model"])
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.title("Performance Comparison of Machine Learning Models")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "paper_model_metrics.png", dpi=300)
plt.close()


# ---------------- FIGURE 5: ACCURACY COMPARISON ----------------

plt.figure(figsize=(9, 5))

bars = plt.bar(df["Model"], df["Accuracy"])

plt.title("Accuracy Comparison of SVM, Random Forest and KNN")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{bar.get_height():.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(GRAPH_DIR / "paper_accuracy_comparison.png", dpi=300)
plt.close()


# ---------------- COMPLETED ----------------

print("\nALL GRAPHS GENERATED SUCCESSFULLY!")

print("\nNew figures saved in:", GRAPH_DIR)

for file in sorted(GRAPH_DIR.glob("paper_*.png")):
    print(file.name)
