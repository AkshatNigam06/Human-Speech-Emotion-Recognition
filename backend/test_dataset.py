import os

dataset_path = "../dataset/Audio_Speech_Actors_01-24"

count = 0

for actor in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                count += 1

print("Total audio files:", count)