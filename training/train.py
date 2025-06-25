import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

def build_feature_vector(row):
    return np.concatenate([
        row['mfcc_mean'],
        row['mfcc_std'],
        row['chroma_mean'],
        row['chroma_std'],
        [row['spec_cent_mean']],
        [row['spec_cent_std']],
        [row['is_song']]
    ])

def main():
    print("Starting emotion classification training...")

    try:
        print("\nCreating metadata...")
        metadata = utils.create_metadata(DATA_DIR)
        print(f"Found {len(metadata)} valid audio files")

        print("\nExtracting features (no augmentation)...")
        features = []
        for idx, row in metadata.iterrows():
            feat = utils.extract_features(
                filepath=row['filepath'],
                vocal_channel=row['vocal_channel'],
                emotion_code=row['emotion'],
                augment=False
            )
            if feat:
                features.append(feat)

        features_df = pd.DataFrame(features)
        print(f"\nExtracted features from {len(features_df)} files")

        X = np.array([build_feature_vector(pd.Series(row)) for _, row in features_df.iterrows()])
        le = LabelEncoder()
        y = le.fit_transform(features_df['emotion'])

        X_train, X_test, y_train, y_test, df_train, _ = train_test_split(
            X, y, features_df, test_size=0.2, stratify=y, random_state=42, shuffle=True
        )

        print("\nAugmenting training data...")
        aug_X, aug_y = [], []
        for idx, row in df_train.iterrows():
            feat = utils.extract_features(
                filepath=row['filepath'],
                vocal_channel=row['vocal_channel'],
                emotion_code=row['emotion'],
                augment=True
            )
            if feat:
                aug_X.append(build_feature_vector(pd.Series(feat)))
                aug_y.append(y_train[idx])
        X_train = np.vstack([X_train, aug_X])
        y_train = np.concatenate([y_train, aug_y])
        print(f"Training data after augmentation: {X_train.shape[0]} samples")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.h5'), save_best_only=True)
        ]

        print("\nTraining model...")
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=200, batch_size=32,
                            callbacks=callbacks, verbose=1)

        print("\nEvaluating model...")
        model.load_weights(os.path.join(MODEL_DIR, 'best_model.h5'))
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")

        y_pred = np.argmax(model.predict(X_test), axis=1)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        
        f1_scores = f1_score(y_test, y_pred, average=None)
        for i, label in enumerate(le.classes_):
            print(f"{label}: {f1_scores[i]:.4f}")

        print("\nGenerating confusion matrix...")
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
        plt.close()

        print("\nSaving model and artifacts...")
        model.save(os.path.join(MODEL_DIR, 'emotion_classifier.h5'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

        print("\nTraining complete.")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import time
    start = time.time()
    print(f"Start: {time.ctime(start)}")
    main()
    end = time.time()
    print(f"End: {time.ctime(end)}")
    print(f"Duration: {(end - start)/60:.2f} minutes")
