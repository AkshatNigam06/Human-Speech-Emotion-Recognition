
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "Audio_Speech_Actors_01-24"
GRAPH_DIR = PROJECT_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

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
axes = axes.flatten()

for ax, (emotion, code) in zip(axes, emotion_codes.items()):

    audio_files = sorted(DATASET_DIR.glob(f"Actor_*/*-*-{code}-*-*-*-*.wav"))

    if not audio_files:
        ax.set_title(f"{emotion} - Audio not found")
        ax.axis("off")
        continue

    audio, sr = librosa.load(audio_files[0], sr=None)

    time = librosa.times_like(
        librosa.feature.rms(y=audio)[0],
        sr=sr
    )

    ax.plot(
        [i / sr for i in range(len(audio))],
        audio,
        linewidth=0.5
    )

    ax.set_title(f"{emotion} Speech Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)

plt.suptitle(
    "Speech Waveform Comparison Across Eight Emotions",
    fontsize=16
)

plt.tight_layout()

output = GRAPH_DIR / "emotion_waveform_comparison.png"

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print("Figure 2 saved:", output)
