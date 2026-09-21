from feature_extraction import extract_mfcc

file_path = "../dataset/Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav"

features = extract_mfcc(file_path)

print("MFCC shape:", features.shape)
print("MFCC features:", features)