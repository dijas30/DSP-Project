import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
from dsp_features import extract_features

def prepare_dataset(dataset_path):
    X, y = [], []
    speakers = sorted(os.listdir(dataset_path))
    for speaker in speakers:
        speaker_dir = os.path.join(dataset_path, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for file in os.listdir(speaker_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_dir, file)
                try:
                    y_audio, sr = librosa.load(file_path, sr=16000, mono=True)
                    feats = extract_features(y_audio, sr)
                    # Skip features with NaN or Inf values
                    if not np.isnan(feats).any() and not np.isinf(feats).any():
                        X.append(feats)
                        y.append(speaker)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    X = np.array(X)
    y = np.array(y)
    print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features.")
    return X, y

def train_model(dataset_path, save_path="speaker_model.joblib", test_size=0.2):
    print("Extracting features...")
    X, y = prepare_dataset(dataset_path)
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=600,
                                   max_depth=25,
                                   min_samples_split=4,
                                   min_samples_leaf=3,
                                   max_features='sqrt',
                                   bootstrap=True,
                                   random_state=42)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"🎯 Accuracy: {acc*100:.2f}%")
    joblib.dump({"model": model, "scaler": scaler, "accuracy": acc, "X_train": X_train, "y_train": y_train}, save_path)
    print(f"Model saved as {save_path}")
    return model, scaler, acc, X_train, y_train

def load_model(model_path):
    data = joblib.load(model_path)
    model = data["model"]
    scaler = data["scaler"]
    accuracy = data.get("accuracy", None)
    X_train = data.get("X_train", None)
    y_train = data.get("y_train", None)
    return model, scaler, accuracy, X_train, y_train

def predict_speaker(file_path, model, scaler):
    y_audio, sr = librosa.load(file_path, sr=16000, mono=True)
    features = extract_features(y_audio, sr)
    if np.isnan(features).any():
        return None, y_audio, sr, features
    features_scaled = scaler.transform(features.reshape(1, -1))
    return model.predict(features_scaled)[0], y_audio, sr, features