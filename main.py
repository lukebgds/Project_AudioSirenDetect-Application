import os
import time
from pathlib import Path
from colorama import Fore, Style
from spectrogram_generator import generate_mel_spectrogram
from model_predictor import load_trained_model, predict_image
from audio_capture import record_segment

SAMPLE_RATE = 22050
DURATION    = 4
CHANNELS    = 1

# Caminho do modelo (constante)
MODEL_PATH  = Path("model/cnn_categorical_model_Final.h5")

def display_dashboard(recording_count, last_predictions):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(Fore.CYAN + "==================== RELATÓRIO ====================" + Style.RESET_ALL)
    print(f"Total de gravações: {recording_count}")
    print(Fore.YELLOW + "\nÚltimos resultados de detecção:" + Style.RESET_ALL)
    for i, (pred_class, conf, path) in enumerate(last_predictions[-3:], 1):
        cor_classe = Fore.GREEN if pred_class == 'yes_siren' else Fore.YELLOW
        print(f"\nFaixa {i}: {cor_classe}{pred_class}{Style.RESET_ALL}")
        print(f"Confiança: {conf:.2f}")
        print(f"Espectrograma salvo em: {path}")
    print("\n==============================")
    estado    = 'Sirene Detectada' if last_predictions[-1][0] == 'yes_siren' else 'Tudo normal'
    cor_estado = Fore.GREEN if last_predictions[-1][0] == 'yes_siren' else Fore.YELLOW
    print(f"Estado atual: {cor_estado + estado}{Style.RESET_ALL}")
    print(f"Confiança do último segmento: {last_predictions[-1][1]:.2f}")
    print("\nPressione Ctrl+C para interromper a gravação.")

def main():
    if not MODEL_PATH.exists():
        print(f"Modelo não encontrado em {MODEL_PATH}. Certifique-se de que o arquivo .h5 está correto.")
        return

    model = load_trained_model(MODEL_PATH)
    print(f"Modelo carregado com sucesso de {MODEL_PATH}.")

    recording_name = input("Digite o nome da gravação: ").strip()
    base_dir = Path("samples")
    recording_folder = base_dir / recording_name
    recording_folder.mkdir(parents=True, exist_ok=True)

    print("Iniciando gravação. Pressione Ctrl+C para parar.\n")

    recording_count = 1
    last_predictions = []

    try:
        while True:
            save_path = recording_folder / f"{recording_name}_spec_{recording_count}.png"

            print(f"Gravando segmento {recording_count}...")
            audio_data = record_segment(SAMPLE_RATE, DURATION, CHANNELS)

            generate_mel_spectrogram(audio_data, SAMPLE_RATE, save_path)
            print(f"Espectrograma salvo em {save_path}")

            predicted_class, confidence = predict_image(model, save_path)
            last_predictions.append((predicted_class, confidence, save_path))
            display_dashboard(recording_count, last_predictions)

            recording_count += 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nGravação interrompida pelo usuário.")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()