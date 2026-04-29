import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import customtkinter as ctk
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import hilbert
from dsp_features import formant_frequencies
from model_utils import train_model, load_model, predict_speaker
from config import DATA_FOLDER, MODEL_FOLDER, MODEL_PATH

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SpeakerRecognizerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(" Speaker Recognizer")
        self.geometry("1400x900")
        self.configure(bg="#1e1e1e")
        self.model, self.scaler, self.accuracy, self.X_train, self.y_train = None, None, None, None, None
        self.audio_file = None

        top_frame = ctk.CTkFrame(self, fg_color="#2e2e2e")
        top_frame.pack(fill="x", pady=10, padx=10)

        self.train_btn = ctk.CTkButton(top_frame, text="Train Model", command=self.train_model_ui)
        self.train_btn.pack(side="left", padx=10)

        self.load_model_btn = ctk.CTkButton(top_frame, text="Load Model", command=self.load_model_ui)
        self.load_model_btn.pack(side="left", padx=10)

        self.select_file_btn = ctk.CTkButton(top_frame, text="Select Audio File", command=self.select_audio)
        self.select_file_btn.pack(side="left", padx=10)

        self.predict_btn = ctk.CTkButton(top_frame, text="Predict & Show Features", command=self.predict_and_plot)
        self.predict_btn.pack(side="left", padx=10)

        self.show_more_btn = ctk.CTkButton(top_frame, text="Show More Features", command=self.show_more_features)
        self.show_more_btn.pack(side="left", padx=10)

        self.model_status = ctk.CTkLabel(top_frame, text="Model: Not loaded")
        self.model_status.pack(side="left", padx=20)
        self.file_status = ctk.CTkLabel(top_frame, text="File: Not selected")
        self.file_status.pack(side="left", padx=20)

        self.pred_label = ctk.CTkLabel(self, text="Prediction: None", font=("Helvetica", 18, "bold"), text_color="#FFD700")
        self.pred_label.pack(pady=10)

        self.tab_control = ttk.Notebook(self)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab4 = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab1, text='Audio Features')
        self.tab_control.add(self.tab2, text='Additional Features')
        self.tab_control.add(self.tab3, text='Dataset & Model Info')
        self.tab_control.add(self.tab4, text='Prediction Comparison')
        self.tab_control.pack(expand=1, fill='both', padx=10, pady=10)

        # Tab1: Audio plots
        self.fig, self.axs = plt.subplots(3, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor("#1e1e1e")
        for ax in self.axs.flat:
            ax.set_facecolor("#2e2e2e")
            ax.tick_params(colors="white", labelcolor="white")
        self.fig.tight_layout(pad=3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab1)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Tab3: Dataset info
        self.info_text = tk.Text(self.tab3, bg="#2e2e2e", fg="white", font=("Helvetica", 12))
        self.info_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.update_info_text("Load or train a model to view dataset info.\n")

    def update_info_text(self, text):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)

    # Training 
    def train_model_ui(self):
        self.update_info_text(" Training model... Please wait.\n")
        self.update()
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        model, scaler, acc, X_train, y_train = train_model(DATA_FOLDER, save_path=MODEL_PATH)
        self.model, self.scaler, self.accuracy, self.X_train, self.y_train = model, scaler, acc, X_train, y_train
        self.model_status.configure(text=f"Model trained ✔ | Accuracy: {acc*100:.2f}%")
        self.update_info_text(f" Training complete!\nAccuracy: {acc*100:.2f}%\nSamples: {X_train.shape[0]}\nFeatures: {X_train.shape[1]}")

    # Load Model 
    def load_model_ui(self):
        model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Joblib files", "*.joblib")], initialdir=MODEL_FOLDER)
        if model_path:
            self.model, self.scaler, self.accuracy, self.X_train, self.y_train = load_model(model_path)
            acc_text = f" | Accuracy: {self.accuracy*100:.2f}%" if self.accuracy else ""
            self.model_status.configure(text=f"Model: {os.path.basename(model_path)}{acc_text}")
            self.update_info_text(f"Model loaded from:\n{model_path}\nAccuracy: {self.accuracy*100:.2f}%")

    # Select Audio 
    def select_audio(self):
        file_path = filedialog.askopenfilename(title="Select WAV File", filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.audio_file = file_path
            self.file_status.configure(text=f"File: {os.path.basename(file_path)}")

    # Prediction & plotting 
    def predict_and_plot(self):
        if not self.model or not self.audio_file:
            self.pred_label.configure(text="Prediction: Load model & file first", text_color="#FF4500")
            return
        speaker, y_audio, sr, features = predict_speaker(self.audio_file, self.model, self.scaler)
        if speaker is None:
            self.pred_label.configure(text="Feature extraction failed", text_color="#FF4500")
            return
        self.pred_label.configure(text=f"Prediction: {speaker}", text_color="#00FF00")

        # Tab1 plots
        for ax in self.axs.flat:
            ax.clear()
            ax.set_facecolor("#2e2e2e")
            ax.tick_params(colors="white", labelcolor="white")
        librosa.display.waveshow(y_audio, sr=sr, ax=self.axs[0,0], color="#1f77b4")
        self.axs[0,0].set_title("Waveform", color="white")
        S = librosa.stft(y_audio)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=self.axs[0,1], cmap="magma")
        self.axs[0,1].set_title("Spectrogram", color="white")
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=self.axs[1,0], cmap="cool")
        self.axs[1,0].set_title("MFCC", color="white")
        chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr)
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=self.axs[1,1], cmap="viridis")
        self.axs[1,1].set_title("Chroma", color="white")
        analytic_signal = hilbert(y_audio)
        amplitude_envelope = np.abs(analytic_signal)
        self.axs[2,0].plot(amplitude_envelope, color="#ff7f0e")
        self.axs[2,0].set_title("Amplitude Envelope", color="white")
        rms = librosa.feature.rms(y=y_audio)[0]
        self.axs[2,1].plot(rms, color="#2ca02c")
        self.axs[2,1].set_title("RMS Energy", color="white")
        self.fig.tight_layout(pad=3)
        self.canvas.draw()

        # Tab4: Prediction comparison
        self.visualize_prediction_comparison(speaker, features)

    # Prediction Comparison (Enhanced) 
    def visualize_prediction_comparison(self, speaker, test_features):
        if self.X_train is None or self.y_train is None:
            return

        import seaborn as sns
        from sklearn.metrics.pairwise import cosine_similarity

        speaker_idx = np.where(self.y_train == speaker)[0]
        speaker_features = self.X_train[speaker_idx]
        mean_speaker_features = np.mean(speaker_features, axis=0)
        feature_diff = test_features - mean_speaker_features
        similarity_scores = cosine_similarity(test_features.reshape(1, -1), speaker_features)[0]
        most_similar_idx = np.argmax(similarity_scores)

        for widget in self.tab4.winfo_children():
            widget.destroy()

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axs.flat:
            ax.set_facecolor("#2e2e2e")
            ax.tick_params(colors="white", labelcolor="white")

        axs[0,0].plot(mean_speaker_features, label=f"Mean {speaker}", color="cyan")
        axs[0,0].plot(test_features, label="Test Audio", color="magenta")
        axs[0,0].set_title("Mean vs Test Features", color="white")
        axs[0,0].legend()

        axs[0,1].scatter(mean_speaker_features, test_features, c="yellow", edgecolors="white")
        axs[0,1].plot([mean_speaker_features.min(), mean_speaker_features.max()],
                      [mean_speaker_features.min(), mean_speaker_features.max()], 
                      color="white", linestyle="--")
        axs[0,1].set_title("Scatter: Mean vs Test", color="white")
        axs[0,1].set_xlabel("Mean Speaker Feature")
        axs[0,1].set_ylabel("Test Feature")

        axs[1,0].bar(range(len(feature_diff)), feature_diff, color="orange")
        axs[1,0].set_title("Per-feature Differences", color="white")
        axs[1,0].set_xlabel("Feature Index")
        axs[1,0].set_ylabel("Difference")

        similarity_matrix = similarity_scores.reshape(1, -1)
        sns.heatmap(similarity_matrix, cmap="viridis", ax=axs[1,1], cbar=True, annot=True, fmt=".2f")
        axs[1,1].set_title(f"Similarity Heatmap (Most Similar Sample: {most_similar_idx})", color="white")
        axs[1,1].set_xlabel("Sample Index")
        axs[1,1].set_ylabel("Similarity")
        axs[1,1].add_patch(plt.Rectangle((most_similar_idx, 0), 1, 1, fill=False, edgecolor="red", lw=2))

        for ax in axs.flat:
            for spine in ax.spines.values():
                spine.set_color("white")

        fig.tight_layout(pad=3)
        canvas = FigureCanvasTkAgg(fig, master=self.tab4)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    # Additional Features 
    def show_more_features(self):
        if not self.audio_file:
            return
        self.tab_control.select(self.tab2)
        y_audio, sr = librosa.load(self.audio_file, sr=16000, mono=True)
        fig2, axs2 = plt.subplots(3, 2, figsize=(12, 8))
        fig2.patch.set_facecolor("#1e1e1e")
        for ax in axs2.flat:
            ax.set_facecolor("#2e2e2e")
            ax.tick_params(colors="white", labelcolor="white")
        pitches, magnitudes = librosa.piptrack(y=y_audio, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0] if len(pitch_values) > 0 else np.array([0])
        axs2[0,0].plot(pitch_values, color="#ff69b4")
        axs2[0,0].set_title("Pitch Contour", color="white")
        formants = formant_frequencies(y_audio, sr)
        axs2[0,1].bar(range(1,len(formants)+1), formants, color="#ffa500")
        axs2[0,1].set_title("Formant Frequencies", color="white")
        centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sr)[0]
        axs2[1,0].plot(centroid, color="#00ced1")
        axs2[1,0].set_title("Spectral Centroid", color="white")
        bandwidth = librosa.feature.spectral_bandwidth(y=y_audio, sr=sr)[0]
        axs2[1,1].plot(bandwidth, color="#adff2f")
        axs2[1,1].set_title("Spectral Bandwidth", color="white")
        rolloff = librosa.feature.spectral_rolloff(y=y_audio, sr=sr)[0]
        axs2[2,0].plot(rolloff, color="#ff4500")
        axs2[2,0].set_title("Spectral Rolloff", color="white")
        rms = librosa.feature.rms(y=y_audio)[0]
        axs2[2,1].plot(rms, color="#9400d3")
        axs2[2,1].set_title("RMS Energy", color="white")
        fig2.tight_layout(pad=3)
        for widget in self.tab2.winfo_children():
            widget.destroy()
        canvas2 = FigureCanvasTkAgg(fig2, master=self.tab2)
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        canvas2.draw()