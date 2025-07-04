import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import librosa
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_DIR = 'model'
SAMPLE_RATE = 22050
DURATION = 3  

def load_resources():
    """Load model and artifacts"""
    print("Loading model and artifacts...")
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'emotion_classifier.h5'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print(f"Model loaded. Classes: {le.classes_}")
    return model, scaler, le

def extract_features(audio_path, vocal_channel=1):
    """Extract features from audio file"""
    try:
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast')

        target_length = SAMPLE_RATE * DURATION
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))
        else:
            signal = signal[:target_length]

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
            'is_song': 1 if vocal_channel == 2 else 0
        }

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def test_single_file(audio_path, model, scaler, le, vocal_channel=1, actual_emotion=None):
    """Test model on a single audio file"""
    print(f"\nTesting file: {os.path.basename(audio_path)}")

    features = extract_features(audio_path, vocal_channel)
    if not features:
        return None

    x = np.concatenate([
        features['mfcc_mean'],
        features['mfcc_std'],
        features['chroma_mean'],
        features['chroma_std'],
        [features['spec_cent_mean']],
        [features['spec_cent_std']],
        [features['is_song']]
    ]).reshape(1, -1)

    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled, verbose=0)
    pred_prob = tf.nn.softmax(pred).numpy()[0]

    pred_class = np.argmax(pred_prob)
    predicted_emotion = le.classes_[pred_class]

    print(f"Predicted emotion: {predicted_emotion} ({pred_prob[pred_class]:.2%} confidence)")
    if actual_emotion:
        print(f"Actual emotion: {actual_emotion}")

    print("\nEmotion Probabilities:")
    for i, prob in enumerate(pred_prob):
        print(f"- {le.classes_[i]}: {prob:.2%}")

    return predicted_emotion

def batch_test(test_dir, model, scaler, le):
    """Test model on all audio files in a directory"""
    results = []

    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                parts = file.split('-')
                vocal_channel = int(parts[1]) if len(parts) > 1 else 1
                actual_emotion_code = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
                actual_emotion = le.classes_[actual_emotion_code - 1] if actual_emotion_code else None

                features = extract_features(filepath, vocal_channel)
                if not features:
                    continue

                x = np.concatenate([
                    features['mfcc_mean'],
                    features['mfcc_std'],
                    features['chroma_mean'],
                    features['chroma_std'],
                    [features['spec_cent_mean']],
                    [features['spec_cent_std']],
                    [features['is_song']]
                ]).reshape(1, -1)

                x_scaled = scaler.transform(x)
                pred = model.predict(x_scaled, verbose=0)
                pred_class = np.argmax(pred, axis=1)[0]
                predicted_emotion = le.classes_[pred_class]

                results.append({
                    'filename': file,
                    'predicted': predicted_emotion,
                    'actual': actual_emotion,
                    'correct': actual_emotion == predicted_emotion if actual_emotion else None
                })

    return pd.DataFrame(results)

def analyze_results(results_df, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "test_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    if 'actual' in results_df.columns and 'correct' in results_df.columns:
        accuracy = results_df['correct'].mean()
        print(f"\nAccuracy: {accuracy:.2%}")

    if 'actual' in results_df.columns:
        report = classification_report(results_df['actual'], results_df['predicted'])
        print("\nClassification Report:\n", report)
        with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        cm = confusion_matrix(results_df['actual'], results_df['predicted'], labels=le.classes_)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        print("Confusion matrix saved.")

    # Prediction distribution
    plt.figure(figsize=(10, 5))
    results_df['predicted'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Prediction Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
    plt.close()
    print("Prediction distribution saved.")

    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Speech Emotion Recognition Model')
    parser.add_argument('--file', type=str, help='Path to a single audio file to test')
    parser.add_argument('--dir', type=str, help='Directory containing audio files for batch testing')
    parser.add_argument('--vocal', type=int, default=1, choices=[1, 2],
                        help='Vocal channel (1=speech, 2=song)')
    parser.add_argument('--actual', type=str, help='Actual emotion (for single file test)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')

    args = parser.parse_args()
    model, scaler, le = load_resources()

    if args.file:
        test_single_file(args.file, model, scaler, le, args.vocal, args.actual)
    elif args.dir:
        results_df = batch_test(args.dir, model, scaler, le)
        if not results_df.empty:
            analyze_results(results_df, args.output)
        else:
            print("No valid audio files found.")
    else:
        print("Specify either --file or --dir for testing.")
        print("Example: python test_model.py --file sample.wav --actual happy")
