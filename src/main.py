import os
import time
import sounddevice as sd
from pathlib import Path
from collections import deque
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

def clear_console():
    """Limpa o console dependendo do sistema operacional."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_detection_queue(detection_queue):
    """Formata a fila de detecção para exibição com cores."""
    formatted = []
    for detection in detection_queue:
        if detection:
            formatted.append(f"{Fore.GREEN}True{Style.RESET_ALL}")
        else:
            formatted.append(f"{Fore.YELLOW}False{Style.RESET_ALL}")
    return formatted

def display_dashboard(detection_queue, confidence, current_status):
    """Exibe o dashboard com a fila de detecção, confiança e estado atual."""
    clear_console()
    formatted_queue = format_detection_queue(detection_queue)
    print(f"{Fore.CYAN}======================= RELATÓRIO ======================={Style.RESET_ALL}")
    print(f"{Fore.WHITE}Confiança (média do último segmento): {confidence:.2f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN if current_status == 'Sirene Detectada!' else Fore.YELLOW}Estado Atual: {current_status}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Resultados das últimas 3 detecções: [{', '.join(formatted_queue)}]{Style.RESET_ALL}")
    print(f"{Fore.CYAN}========================================================={Style.RESET_ALL}")

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

    # Fila para armazenar as últimas 3 detecções
    detection_queue = deque(maxlen=3)

    recording_count = 1

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

            # Verifica se a classe detectada é uma sirene
            is_siren = predicted_class == 'yes_siren'

            # Atualiza a fila com a nova detecção
            detection_queue.append(is_siren)

            # Validação com base nas últimas 3 detecções
            current_status = "Sirene Detectada!" if detection_queue.count(True) >= 2 else "Tudo Normal..."

            # Atualiza o dashboard
            display_dashboard(detection_queue, confidence, current_status)

            recording_count += 1
            time.sleep(1)  # Pequena pausa para a interface ser legível

    except KeyboardInterrupt:
        print("Gravação interrompida pelo usuário.")
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
