
import librosa
import numpy as np

SAMPLE_RATE = 22050
N_MFCC = 40


def extract_mfcc(file_path):

    # Load audio
    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    # First-order Delta
    delta_mfcc = librosa.feature.delta(mfcc)

    # Second-order Delta
    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Calculate statistical features
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)

    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)

    # Combine all features
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        delta_std,
        delta2_mean,
        delta2_std
    ])

    return features.astype(np.float32)
