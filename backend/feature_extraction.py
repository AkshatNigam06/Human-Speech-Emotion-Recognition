import librosa
import numpy as np


def extract_mfcc(file_path):

    audio, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc