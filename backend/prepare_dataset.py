import os
import numpy as np
from feature_extraction import extract_mfcc


dataset_path = "../dataset/Audio_Speech_Actors_01-24"

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


for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                file_path = os.path.join(actor_path, file)

                parts = file.split("-")
                emotion_code = parts[2]
                emotion = emotion_labels[emotion_code]

                features = extract_mfcc(file_path)

                X.append(features)
                y.append(emotion)


X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
np.save("../dataset/X.npy", X)
np.save("../dataset/y.npy", y)

print("Dataset saved successfully!")