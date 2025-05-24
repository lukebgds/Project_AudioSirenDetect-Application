import os
import time
from pathlib import Path
from colorama import Fore, Style
from spectrogram_generator import generate_mel_spectrogram
from model_predictor import load_trained_model, predict_image
from audio_capture import record_segment

SAMPLE_RATE = 22050
DURATION = 4
CHANNELS = 1

# Caminho do modelo
MODEL_PATH  = Path("model/modelo_multiclasse_siren.h5")

def display_dashboard(recording_count, last_predictions):
    # Limpa o console (funciona tanto no Windows quanto em sistemas Unix)
    os.system('cls' if os.name == 'nt' else 'clear')

    # Cabeçalho do relatório
    print(Fore.CYAN + "====================== RELATÓRIO ======================" + Style.RESET_ALL)
    print(f"Total de segmentos processados: {recording_count}\n")

    # Exibe, de forma tabular, as últimas 3 predições (ou menos, se ainda não houver 3)
    print(Fore.YELLOW + "Últimos resultados de detecção:" + Style.RESET_ALL)
    print("-" * 60)
    print(f"{'Índice':<8}{'Classe':<15}{'Confiança':<12}{'Arquivo (espectrograma)'}")
    print("-" * 60)
    for idx, (pred_class, conf, path) in enumerate(last_predictions[-3:], start=max(1, recording_count - len(last_predictions) + 1)):
        cor_classe = Fore.GREEN if pred_class == 'yes_siren' else Fore.YELLOW
        classe_exib = f"{cor_classe}{pred_class}{Style.RESET_ALL}"
        print(f"{idx:<8}{classe_exib:<15}{conf:<12.2f}{path.name}")
    print("-" * 60)

    # Estado atual (baseado no último segmento)
    if last_predictions:
        ultima_classe, ultima_conf, _ = last_predictions[-1]
        estado_texto = "Sirene Detectada" if ultima_classe == 'yes_siren' else "Tudo normal"
        estado_cor   = Fore.GREEN if ultima_classe == 'yes_siren' else Fore.YELLOW

        print(f"\nEstado atual: {estado_cor + estado_texto}{Style.RESET_ALL}")
        print(f"Confiança do último segmento: {ultima_conf:.2f}")
    else:
        print("\nAinda não há predições para exibir.")

    print("\n(Pressione Ctrl+C para interromper a gravação.)")

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
