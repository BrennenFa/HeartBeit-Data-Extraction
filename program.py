import wfdb
import scipy.io
import numpy as np
import glob
import os
from scipy.signal import resample

def load_ecg_record(record_path):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal  # shape: (time, channels)
    return signal



def preprocess_signal(signal, original_fs=500, target_fs=250, duration=10):
    signal = signal[:original_fs * duration]  # clip to 10s
    signal = resample(signal, int(target_fs * duration))
    signal = (signal - np.mean(signal)) / np.std(signal)
    return signal  # shape: (time, channels)


def create_sequence_windows(signal, window_size=250, step=125):
    windows = []
    for start in range(0, signal.shape[0] - window_size + 1, step):
        window = signal[start:start+window_size]
        windows.append(window)
    return np.stack(windows)  # shape: (seq_len, window_size, channels)

import torch
import torch.nn as nn

class ECGTransformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=2, seq_len=80):
        super().__init__()
        self.input_proj = nn.Linear(250 * input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, model_dim)  # Feature vector output

    def forward(self, x):
        # x shape: (batch, seq_len, window_size, channels)
        batch_size, seq_len, win_size, channels = x.size()
        x = x.view(batch_size, seq_len, -1)  # flatten each window
        x = self.input_proj(x)               # (batch, seq_len, model_dim)
        x = x.permute(1, 0, 2)               # (seq_len, batch, model_dim)
        features = self.transformer(x)       # (seq_len, batch, model_dim)
        return features.permute(1, 0, 2)     # (batch, seq_len, model_dim)



def extract_ecg_features(record_path):
    signal = load_ecg_record(record_path)
    signal = preprocess_signal(signal)
    sequence = create_sequence_windows(signal)
    
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # batch=1
    model = ECGTransformer(input_dim=signal.shape[1])
    features = model(sequence_tensor)
    
    # Optionally, pool features
    pooled = features.mean(dim=1)  # global average pooling across time
    np.save('features.npy', pooled.numpy())

    return pooled.detach().numpy()


if __name__ == '__main__':
    files = glob.glob('data/**/*.dat', recursive=True)
    print(files)
    print("FILES  ^")
    for file in files:
        record_path = os.path.splitext(file)[0]  # strip .mat extension
        print(f"Processing: {record_path}")
        try:
            features = extract_ecg_features(record_path)
            print(f"Extracted features shape: {features.shape}")
        except Exception as e:
            print(f"Failed to process {record_path}: {e}")
