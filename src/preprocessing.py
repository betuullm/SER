import os
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr

# Kayıtların bulunduğu ana dizin
data_root = r"d:\SesVeriSeti"
# Önişlenmiş dosyaların kaydedileceği dizin
output_root = os.path.join(data_root, "preprocessed4")
os.makedirs(output_root, exist_ok=True)

def preprocess_audio(file_path, out_path, vad_db=30, vad_margin_sec=1.0, vad_tail_sec=0.7):
    # Ses dosyasını yükle
    y, sr = librosa.load(file_path, sr=None)
    # Gürültü azaltma
    y = nr.reduce_noise(y=y, sr=sr)
    # VAD ile aktif sesli bölgeyi bul
    intervals = librosa.effects.split(y, top_db=vad_db)
    if len(intervals) > 0:
        # Başlangıç ve bitişi belirle, baştan ve sondan margin bırak
        start = max(intervals[0][0] - int(vad_margin_sec * sr), 0)
        # Son aktif sesin bitişinden sonra tail kadar bırak
        end = min(intervals[-1][1] + int(vad_tail_sec * sr), len(y))
        y = y[start:end]
    # Normalizasyon
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    sf.write(out_path, y, sr)

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
