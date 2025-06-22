import librosa
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

def extract_features(filepath, vocal_channel, emotion_code, duration=3, sr=22050):
    try:
        signal, sr = librosa.load(filepath, sr=sr, duration=duration)
        if len(signal) < sr * duration:
            signal = np.pad(signal, (0, sr * duration - len(signal)))
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        
        return {
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'spec_cent_mean': np.mean(spectral_centroid),
            'spec_cent_std': np.std(spectral_centroid),
            'is_song': 1 if vocal_channel == 2 else 0,
            'emotion': emotion_labels[emotion_code]
        }
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None