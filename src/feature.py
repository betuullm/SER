import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {}
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc)
        features[f"mfcc_{i+1}_std"] = np.std(mfcc)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    features["mel_mean"] = np.mean(mel)
    features["mel_std"] = np.std(mel)
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["contrast_mean"] = np.mean(contrast)
    features["contrast_std"] = np.std(contrast)
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
