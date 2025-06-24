import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {}
    # MFCC (13 için mean ve std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc)
        features[f"mfcc_{i+1}_std"] = np.std(mfcc)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, c in enumerate(chroma):
        features[f"chroma_{i+1}_mean"] = np.mean(c)
        features[f"chroma_{i+1}_std"] = np.std(c)
    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features["mel_mean"] = np.mean(mel)
    features["mel_std"] = np.std(mel)
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i, sc in enumerate(contrast):
        features[f"contrast_{i+1}_mean"] = np.mean(sc)
        features[f"contrast_{i+1}_std"] = np.std(sc)
    # RMS (energy)
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["centroid_mean"] = np.mean(centroid)
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["rolloff_mean"] = np.mean(rolloff)
    return features

def extract_features_from_folder(folder_path):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                feats = extract_features(file_path)
                feats['file'] = file
                feats['folder'] = os.path.basename(root)
                data.append(feats)
    return pd.DataFrame(data)

if __name__ == "__main__":
    train_folder = r"D:\Data\Train"
    df = extract_features_from_folder(train_folder)
    df.to_csv("train_features.csv", index=False)
    print("Özellik çıkarımı tamamlandı. train_features.csv kaydedildi.")
