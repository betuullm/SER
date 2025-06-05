import os
import pandas as pd
from glob import glob

def load_labels(label_path):
    df = pd.read_csv(label_path)
    # Deney numarasını string olarak anahtar yap
    label_dict = {str(row['Deney']): row['Label'] for _, row in df.iterrows()}
    return label_dict

def load_dataset(root_dir, label_dict):
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            # Paradigma türünü belirle
            if "birinci" in folder:
                target_list = train_files
                target_labels = train_labels
            elif "ikinci" in folder:
                target_list = test_files
                target_labels = test_labels
            else:
                continue

            # Ses dosyalarını bul (ör: .wav uzantılı)
            audio_files = glob(os.path.join(folder_path, "*.wav"))
            for audio_file in audio_files:
                # Dosya adından deney numarasını çıkar (ör: 001.wav -> 1)
                base = os.path.basename(audio_file)
                deney_no = os.path.splitext(base)[0].lstrip("0")
                label = label_dict.get(deney_no)
                target_list.append(audio_file)
                target_labels.append(label)
    return (train_files, train_labels), (test_files, test_labels)

if __name__ == "__main__":
    root = r"d:\SesVeriSeti"
    label_path = r"c:\Users\Lenovo\Desktop\SER\Labels.csv"
    label_dict = load_labels(label_path)
    (train_files, train_labels), (test_files, test_labels) = load_dataset(root, label_dict)
    print(f"Train dosya sayısı: {len(train_files)}")
    print(f"Test dosya sayısı: {len(test_files)}")
    print(f"Örnek train dosya ve label: {list(zip(train_files, train_labels))[:3]}")
    print(f"Örnek test dosya ve label: {list(zip(test_files, test_labels))[:3]}")
