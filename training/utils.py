# utils.py

import librosa
import numpy as np
import os
import pandas as pd

# Emotion labels mapping
emotion_labels = {
    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
}

def create_metadata(data_dir):
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                try:
                    parts = file[:-4].split('-')
                    if len(parts) != 7:
                        continue
                    
                    parts = [int(p) for p in parts]
                    
                    if parts[2] < 1 or parts[2] > 8:
                        continue
                        
                    data.append({
                        'filepath': os.path.join(root, file),
                        'modality': parts[0],
                        'vocal_channel': parts[1],
                        'emotion': parts[2],
                        'intensity': parts[3],
                        'statement': parts[4],
                        'repetition': parts[5],
                        'actor': parts[6]
                    })
                except (ValueError, IndexError) as e:
                    print(f"Skipping {file}: {str(e)}")
                    continue
    return pd.DataFrame(data)


def add_background_noise(signal, noise_level=0.005):
    noise = np.random.randn(len(signal))
    return signal + noise_level * noise


def extract_features(filepath, vocal_channel, emotion_code, duration=3, sr=22050, augment=False):
    try:
        signal, sr = librosa.load(filepath, sr=sr)
        signal = signal.astype(np.float32)

        # Trim or pad
        target_len = sr * duration
        if len(signal) > target_len:
            signal = signal[:target_len]
        elif len(signal) < target_len:
            signal = np.pad(signal, (0, target_len - len(signal)))

        # Augmentation (only if enabled)
        if augment:
            if np.random.rand() < 0.5:
                rate = np.random.uniform(0.9, 1.1)
                signal = librosa.effects.time_stretch(signal, rate)
            if np.random.rand() < 0.5:
                n_steps = np.random.randint(-2, 3)
                signal = librosa.effects.pitch_shift(signal, sr, n_steps)
            if np.random.rand() < 0.5:
                signal = add_background_noise(signal, noise_level=0.003)

            # Re-adjust length after augmentation
            if len(signal) > target_len:
                signal = signal[:target_len]
            elif len(signal) < target_len:
                signal = np.pad(signal, (0, target_len - len(signal)))

        # Feature extraction
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]

        return {
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'spec_cent_mean': np.mean(spec_cent),
            'spec_cent_std': np.std(spec_cent),
            'is_song': int(vocal_channel == 2),
            'emotion': emotion_labels[emotion_code]
        }

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None
