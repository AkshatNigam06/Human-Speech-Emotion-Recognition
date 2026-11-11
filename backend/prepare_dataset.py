
import os
import numpy as np
from feature_extraction import extract_mfcc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, "../dataset/Audio_Speech_Actors_01-24"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "../dataset")

emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []
actors = []

for actor in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    actor_id = actor.split("_")[-1]

    for file in sorted(os.listdir(actor_path)):

        if not file.lower().endswith(".wav"):
            continue

        parts = file.split("-")

        if len(parts) < 3 or parts[2] not in emotion_labels:
            continue

        file_path = os.path.join(actor_path, file)

        try:
            features = extract_mfcc(file_path)

            X.append(features)
            y.append(emotion_labels[parts[2]])
            actors.append(int(actor_id))

        except Exception as error:
            print(f"Error processing {file}: {error}")

X = np.array(X, dtype=np.float32)
y = np.array(y)
actors = np.array(actors)

if len(X) == 0:
    raise ValueError("No audio files processed!")

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
np.save(os.path.join(OUTPUT_DIR, "actors.npy"), actors)

print("\nDataset prepared successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Actors shape:", actors.shape)
print("Unique actors:", np.unique(actors))
print("Total samples:", len(X))
