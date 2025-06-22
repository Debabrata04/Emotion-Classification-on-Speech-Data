import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Add these imports at the top
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import utils

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# ... [previous code] ...

def main():
    print("Starting emotion classification training...")
    
    try:
        # Step 1: Create metadata
        print("\nCreating metadata...")
        metadata = utils.create_metadata(DATA_DIR)
        print(f"Found {len(metadata)} valid audio files")
        
        # Add emotion labels
        metadata['emotion_label'] = metadata['emotion'].map(utils.emotion_labels)
        print("\nEmotion distribution:")
        print(metadata['emotion_label'].value_counts())
        
        # Step 2: Extract features
        print("\nExtracting features...")
        features = []
        for idx, row in metadata.iterrows():
            feat = utils.extract_features(
                filepath=row['filepath'],
                vocal_channel=row['vocal_channel'],
                emotion_code=row['emotion']
            )
            if feat:
                features.append(feat)
        
        features_df = pd.DataFrame(features)
        print(f"\nSuccessfully extracted features from {len(features_df)} files")
        
        # Step 3: Prepare data
        X = []
        for _, row in features_df.iterrows():
            feature_vector = np.concatenate([
                row['mfcc_mean'],
                row['mfcc_std'],
                row['chroma_mean'],
                row['chroma_std'],
                [row['spec_cent_mean']],
                [row['spec_cent_std']],
                [row['is_song']]
            ])
            X.append(feature_vector)
        
        X = np.array(X)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(features_df['emotion'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print("\nData shapes:")
        print("X_train:", X_train.shape)
        print("X_test:", X_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)
        
        # Step 4: Build and train model
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
            Dense(len(utils.emotion_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.h5'), save_best_only=True)
        ]
        
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Step 5: Evaluate model
        print("\nLoading best model weights...")
        model.load_weights(os.path.join(MODEL_DIR, 'best_model.h5'))
        
        print("Evaluating model...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        
        print("Making predictions...")
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=le.classes_))
        
        # F1 scores
        print("\nCalculating F1 scores...")
        f1_scores = f1_score(y_test, y_pred_classes, average=None)
        print("\nF1 Scores per Class:")
        for i, emotion in enumerate(le.classes_):
            print(f"{emotion}: {f1_scores[i]:.4f}")
        
        # Confusion matrix
        print("\nGenerating confusion matrix...")
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, 
                    yticklabels=le.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
        plt.close()  # Close figure to free memory
        print("Confusion matrix saved")
        
        # Step 6: Save final model and artifacts
        print("\nSaving model artifacts...")
        model.save(os.path.join(MODEL_DIR, 'emotion_classifier.h5'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        print("\nModel and artifacts saved to model/ directory")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTraining failed. See error above.")
    finally:
        print("\nCleaning up resources...")

if __name__ == "__main__":
    import time
    start_time = time.time()
    print(f"Script started at: {time.ctime(start_time)}")
    
    main()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nScript finished at: {time.ctime(end_time)}")
    print(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
