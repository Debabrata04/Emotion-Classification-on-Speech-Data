import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os
import tempfile

def extract_features(audio_path, vocal_channel=1, duration=3, sr=22050):
    try:
        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            temp_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            audio.export(temp_path, format="wav")
            audio_path = temp_path
        
        # Load audio
        signal, sr = librosa.load(audio_path, sr=sr, duration=duration)
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
            'is_song': 1 if vocal_channel == 2 else 0
        }
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        return None
    finally:
        # Clean up temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)