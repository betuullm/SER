import os
import opensmile
import pandas as pd
import torch

def extract_egemaps_features(file_path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_file(file_path)
    return features

def extract_features_from_folder(folder_path):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                feats = extract_egemaps_features(file_path)
                feats = feats.reset_index(drop=True)
                row = feats.iloc[0].to_dict()
                row['file'] = file
                row['folder'] = os.path.basename(root)
                data.append(row)
    return pd.DataFrame(data)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
    folder = r"D:\Data\Train"  # Klasörü ihtiyaca göre değiştirin
    df = extract_features_from_folder(folder)
    print(df.head())
    df.to_csv("egemaps_features.csv", index=False)
