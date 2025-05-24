import os
import time
import sounddevice as sd
from pathlib import Path
from colorama import Fore, Style
from spectrogram_generator import generate_mel_spectrogram
from model_predictor import load_trained_model, predict_image

# Parâmetros de gravação
SAMPLE_RATE = 22050
DURATION = 4
CHANNELS = 1

# Caminho do modelo
MODEL_PATH = Path("modelo_multiclasse_siren.h5")

# Mapeamento de classes
CLASS_NAMES = ['no_siren', 'yes_siren']

def display_dashboard(recording_count, last_predictions):
   
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
    # Carregar o modelo
    if not MODEL_PATH.exists():
        print(f"Modelo não encontrado em {MODEL_PATH}. Certifique-se de que o modelo está na pasta correta.")
        return

    model = load_trained_model(MODEL_PATH)
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
            audio_data = audio_data.flatten()

            # Processa o espectrograma Mel e salva a imagem
            generate_mel_spectrogram(audio_data, SAMPLE_RATE, save_path)
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
