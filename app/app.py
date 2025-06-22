import os
import tempfile
import numpy as np
import tensorflow as tf
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st
from pydub import AudioSegment
from pydub.utils import which

# Must be the first Streamlit command
st.set_page_config(page_title="Speech Emotion Recognition", layout="wide")

# Configure FFmpeg
def setup_ffmpeg():
    """Ensure FFmpeg is available for audio processing."""
    # Check if FFmpeg exists in PATH
    if which("ffmpeg"):
        return True
    
    # Check common installation locations
    common_paths = [
        r"C:\ffmpeg\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            os.environ['PATH'] = os.path.dirname(path) + os.pathsep + os.environ['PATH']
            AudioSegment.converter = path
            return True
    
    return False

if not setup_ffmpeg():
    st.warning("""
    FFmpeg not found! Some audio features may be limited.
    For full functionality:
    1. Download from https://ffmpeg.org/download.html
    2. Extract to C:\\ffmpeg
    3. Add to your system PATH
    """)

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ======== FIX: Import app_utils with proper path handling ========
try:
    import app_utils
except ImportError:
    st.error("Could not import app_utils. Make sure it's in the same directory as app.py")
    st.stop()

# Load resources with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'emotion_classifier.h5'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        return model, scaler, le
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, scaler, le = load_model()

# ======== FIX: Matplotlib compatibility workaround ========
# Set non-interactive backend
mpl.use('Agg')
# Explicitly set the property cycler to avoid attribute errors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

# App UI
st.title("🎤 Speech Emotion Recognition")
st.write("Upload an audio file to detect the emotional state and analyze audio features")

# File uploader with supported formats
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])

if uploaded_file is not None:
    # Audio player
    st.audio(uploaded_file)
    
    # Analysis tabs
    tab1, tab2 = st.tabs(["Emotion Detection", "Audio Analysis"])
    
    with tab1:
        vocal_channel = st.radio("Audio Type:", ("Speech", "Song"), horizontal=True)
        
        if st.button("Detect Emotion"):
            with st.spinner("Processing audio..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        audio_path = tmp.name
                    
                    features = app_utils.extract_features(
                        audio_path, 
                        vocal_channel=2 if vocal_channel == "Song" else 1
                    )
                    
                    if features:
                        x = np.concatenate([
                            features['mfcc_mean'],
                            features['mfcc_std'],
                            features['chroma_mean'],
                            features['chroma_std'],
                            [features['spec_cent_mean']],
                            [features['spec_cent_std']],
                            [features['is_song']]
                        ]).reshape(1, -1)
                        
                        x = scaler.transform(x)
                        pred = model.predict(x)
                        pred_prob = tf.nn.softmax(pred).numpy()[0]
                        emotion_idx = np.argmax(pred_prob)
                        emotion = le.classes_[emotion_idx]
                        
                        st.success(f"**Predicted Emotion:** {emotion}")
                        st.subheader("Emotion Probabilities:")
                        
                        for i, prob in enumerate(pred_prob):
                            st.progress(float(prob), text=f"{le.classes_[i]}: {prob:.2%}")
                        
                        # Plot with version-compatible matplotlib
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.bar(le.classes_, pred_prob, color='skyblue')
                        ax.set_title("Emotion Probability Distribution")
                        ax.set_ylabel("Probability")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.error("Feature extraction failed")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                finally:
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
    
    with tab2:
        st.subheader("Comprehensive Audio Analysis")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_path = tmp.name
            
            y, sr = librosa.load(audio_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Waveform plot - FIXED: using simpler plot method
                st.write("### Waveform")
                fig, ax = plt.subplots(figsize=(8, 3))
                times = np.linspace(0, len(y)/sr, num=len(y))
                ax.plot(times, y)
                ax.set_title('Audio Waveform')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                st.pyplot(fig)
                plt.close()
                
                # Spectral Centroid - FIXED: using simpler plot method
                st.write("### Spectral Centroid")
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                fig, ax = plt.subplots(figsize=(8, 3))
                times = librosa.times_like(cent)
                ax.plot(times, cent.T)
                ax.set(title='Spectral Centroid (Brightness)', xlabel='Time (s)', ylabel='Hz')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Spectrogram plot - FIXED: using pcolormesh instead of specshow
                st.write("### Spectrogram")
                fig, ax = plt.subplots(figsize=(8, 3))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = ax.pcolormesh(librosa.times_like(D), librosa.fft_frequencies(sr=sr), D, 
                                   shading='auto', cmap='viridis')
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                ax.set(title='Spectrogram (Frequency Content)', xlabel='Time (s)', ylabel='Frequency (Hz)')
                ax.set_yscale('log')
                st.pyplot(fig)
                plt.close()
                
                # MFCCs plot - FIXED: using pcolormesh instead of specshow
                st.write("### MFCCs (40 Coefficients)")
                fig, ax = plt.subplots(figsize=(8, 3))
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                img = ax.pcolormesh(librosa.times_like(mfccs), np.arange(mfccs.shape[0]), mfccs, 
                                   shading='auto', cmap='coolwarm')
                fig.colorbar(img, ax=ax)
                ax.set(title='Mel-Frequency Cepstral Coefficients', xlabel='Time (s)', ylabel='MFCC Coefficients')
                st.pyplot(fig)
                plt.close()
            
            # Audio metrics
            st.write("### Audio Characteristics")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Duration", f"{len(y)/sr:.2f} seconds")
                st.metric("Sample Rate", f"{sr} Hz")
            
            with col4:
                rms = librosa.feature.rms(y=y)
                st.metric("Average Volume", f"{np.mean(rms):.4f}")
                st.metric("Max Volume", f"{np.max(np.abs(y)):.4f}")
            
            with col5:
                zcr = librosa.feature.zero_crossing_rate(y)
                st.metric("Zero Crossing Rate", f"{np.mean(zcr):.4f}")
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                st.metric("Tempo Estimate", f"{tempo:.1f} BPM")
                
        except Exception as e:
            st.error(f"Audio analysis failed: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)




