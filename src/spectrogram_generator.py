import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 400

def generate_mel_spectrogram(audio_data, sr, save_path):
 
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(IMAGE_WIDTH / 100, IMAGE_HEIGHT / 100),dpi=100)
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
