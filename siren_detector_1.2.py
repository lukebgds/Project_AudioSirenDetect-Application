import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from colorama import Fore, Style
import time

# VERSÃO BASE COM INTERFACE

# Configurações de áudio
SAMPLE_RATE = 22050  # Taxa de amostragem
DURATION = 4  # Duração de cada segmento de áudio em segundos
CHANNELS = 1  # Áudio mono
IMAGE_WIDTH = 1000  # Largura fixa da imagem em pixels
IMAGE_HEIGHT = 400  # Altura fixa da imagem em pixels

# Classes do modelo
CLASS_NAMES = {0: 'no_siren', 1: 'yes_siren'}

# Caminho do modelo
MODEL_PATH = Path("model/cnn_categorical_model_Final.h5")


def save_mel_spectrogram_with_axes(audio_data, sr, save_path):
    """
    Salva o espectrograma Mel de um segmento de áudio com dimensões fixas de 1000x400 pixels e eixos visíveis.
    """
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Criar o gráfico do espectrograma Mel com eixos visíveis
    fig, ax = plt.subplots(figsize=(IMAGE_WIDTH / 100, IMAGE_HEIGHT / 100), dpi=100)  # Ajusta Dimensões
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')  # Adicionar a barra de cores associada ao espectrograma
    plt.tight_layout()  # Ajustar o layout para evitar sobreposição

    # Salvar a imagem com tamanho exato
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def preprocess_image(img_path):
    """
    Realiza o pré-processamento de uma imagem para a inferência.
    """
    # Carregar e redimensionar a imagem
    img = image.load_img(img_path, target_size=(128, 128))

    # Converter para array NumPy
    img_array = image.img_to_array(img)

    # Normalizar os valores de pixel
    img_array = img_array / 255.0

    # Adicionar a dimensão do batch
    img_final = np.expand_dims(img_array, axis=0)

    return img_final


def predict_image(model, img_path):
    """
    Realiza a predição de classe para uma imagem utilizando o modelo carregado.
    """
    img_final = preprocess_image(img_path)
    predictions = model.predict(img_final)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    return CLASS_NAMES[predicted_class], confidence


def display_dashboard(recording_count, last_predictions):
    """
    Exibe o dashboard atualizado com o estado atual, confiança ao longo do tempo e espectrogramas.
    """
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpa o console

    print(Fore.CYAN + "==================== RELATÓRIO ====================" + Style.RESET_ALL)
    print(f"Total de gravações: {recording_count}")

    # Exibir os resultados das últimas 3 faixas
    print(Fore.YELLOW + "\nÚltimos resultados de detecção:" + Style.RESET_ALL)
    for i, (pred_class, conf, path) in enumerate(last_predictions[-3:], 1):
        print(f"\nFaixa {i}: {Fore.GREEN if pred_class == 'yes_siren' else Fore.YELLOW}{pred_class}{Style.RESET_ALL}")
        print(f"Confiança: {conf:.2f}")
        print(f"Espectrograma salvo em: {path}")

    print("\n==============================")
    print(f"Estado atual: {Fore.GREEN + 'Sirene Detectada' if last_predictions[-1][0] == 'yes_siren' else Fore.YELLOW + 'Tudo normal'}{Style.RESET_ALL}")
    print(f"Confiança do último segmento: {last_predictions[-1][1]:.2f}")
    print("\nPressione Ctrl+C para interromper a gravação.")

def main():
    # Carregar o modelo de CNN
    if not MODEL_PATH.exists():
        print(f"Modelo não encontrado em {MODEL_PATH}. Certifique-se de que o modelo está na pasta correta.")
        return

    model = load_model(MODEL_PATH)
    print(f"Modelo carregado com sucesso de {MODEL_PATH}.")

    # Perguntar o nome da gravação
    recording_name = input("Digite o nome da gravação: ").strip()
    base_dir = Path("samples")
    recording_folder = base_dir / recording_name
    recording_folder.mkdir(parents=True, exist_ok=True)

    print("Iniciando gravação. Pressione Ctrl+C para parar.")

    recording_count = 1
    last_predictions = []  # Para armazenar os resultados das últimas predições

    try:
        while True:
            # Caminho da próxima imagem
            save_path = recording_folder / f"{recording_name}_spec_{recording_count}.png"

            # Captura áudio de 4 segundos
            print(f"Gravando segmento {recording_count}...")
            audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
            sd.wait()  # Espera o fim da gravação

            # Processa o espectrograma Mel e salva a imagem
            save_mel_spectrogram_with_axes(audio_data.flatten(), SAMPLE_RATE, save_path)
            print(f"Espectrograma salvo em {save_path}")

            # Realizar inferência no espectrograma gerado
            predicted_class, confidence = predict_image(model, save_path)

            # Armazenar a predição e a confiança
            last_predictions.append((predicted_class, confidence, save_path))

            # Exibir o dashboard atualizado
            display_dashboard(recording_count, last_predictions)

            recording_count += 1
            time.sleep(1)  # Atraso de 1 segundo para a próxima gravação

    except KeyboardInterrupt:
        print("Gravação interrompida pelo usuário.")
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()
