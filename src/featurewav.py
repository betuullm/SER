import os
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf

def extract_wav2vec_features(file_path, processor, model, device):
    y, sr = sf.read(file_path)
    if len(y.shape) > 1:
        y = y[:, 0]  # stereo ise tek kanala indir
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        input_values = inputs.input_values.to(device)
        if "attention_mask" in inputs:
            attention_mask = inputs.attention_mask.to(device)
            outputs = model(input_values, attention_mask=attention_mask)
        else:
            outputs = model(input_values)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

def extract_features_from_folder_wav2vec(folder_path, processor, model, device):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emb = extract_wav2vec_features(file_path, processor, model, device)
                feats = {f'wav2vec_{i}': emb[i] for i in range(len(emb))}
                feats['file'] = file
                feats['folder'] = os.path.basename(root)
                data.append(feats)
    return pd.DataFrame(data)

if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
    folder = r"D:\SesVeriSeti"  # Klasörü ihtiyaca göre değiştirin
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Çalışma ortamı: {device}")  # Cuda mı cpu mu yazdır
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').to(device)
    df_wav2vec = extract_features_from_folder_wav2vec(folder, processor, model, device)
    df_wav2vec.to_csv("features_wav2vec.csv", index=False)
    print("Wav2vec 2.0 özellik çıkarımı tamamlandı. features_wav2vec.csv kaydedildi.")
