import os
import librosa
import soundfile as sf
import numpy as np

# Kayıtların bulunduğu ana dizin
data_root = r"d:\SesVeriSeti"
# Önişlenmiş dosyaların kaydedileceği dizin
output_root = os.path.join(data_root, "preprocessed")
os.makedirs(output_root, exist_ok=True)

def preprocess_audio(file_path, out_path, sr=16000, top_db=20):
    # Ses dosyasını yükle
    y, orig_sr = librosa.load(file_path, sr=None)
    # Gürültü azaltma (basit spectral gating)
    y = librosa.effects.preemphasis(y)
    # Sessiz bölge kırpma
    y, _ = librosa.effects.trim(y, top_db=top_db)
    # Normalizasyon
    y = y / np.max(np.abs(y))
    # Kaydet (orijinal örnekleme oranı ile)
    sf.write(out_path, y, orig_sr)

# Tüm denek klasörlerini tara
for subject_folder in os.listdir(data_root):
    subject_path = os.path.join(data_root, subject_folder)
    if os.path.isdir(subject_path):
        # Her deneğin önişlenmiş kayıtları kendi klasörüne kaydedilecek
        out_subject_path = os.path.join(output_root, subject_folder)
        os.makedirs(out_subject_path, exist_ok=True)
        count = 0
        for fname in os.listdir(subject_path):
            if fname.endswith('.wav'):
                in_f = os.path.join(subject_path, fname)
                out_f = os.path.join(out_subject_path, fname)
                preprocess_audio(in_f, out_f)
                print(f"Kaydedildi: {out_f}")
                count += 1
        print(f"{subject_folder} klasöründe {count} kayıt kaydedildi.")

print("Tüm ses dosyaları önişlendi ve kaydedildi.")
