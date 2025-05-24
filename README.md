# Project_AudioSirenDetect-Application

## Introdução

Este repositório contém um projeto acadêmico voltado à detecção de sirenes em tempo real utilizando redes neurais convolucionais (CNNs). O sistema foi desenvolvido em Python, com foco na modularização do código e clareza na execução dos processos. Ele captura segmentos de áudio, transforma-os em espectrogramas Mel, e utiliza um modelo treinado para classificar a presença ou ausência de sirenes. 

**Todo o processo é automatizado e apresentado de forma interativa via terminal, com visualização contínua das predições por meio de um dashboard simples.**

> **Observação:** O processo completo de **treinamento do modelo utilizado neste sistema** está documentado separadamente, no repositório complementar:  
> 🔗 [Project_AudioSirenDetect-ModelTraining](https://github.com/lukebgds/Project_AudioSirenDetect-ModelTraining)

## Lógica Geral em Fluxo

1. **Capturar segmentos de áudio** de forma contínua, em intervalos fixos de quatro segundos.  
2. **Converter cada segmento em um espectrograma Mel**, uma representação visual que destaca as características espectrais ao longo do tempo.  
3. **Aplicar um modelo de rede neural convolucional (CNN)** treinado previamente para classificar o espectrograma como contendo ou não uma sirene.  
4. **Exibir um dashboard em console** que atualiza, a cada novo segmento, a quantidade total de gravações, as últimas predições (classe e confiança) e o status atual (“Sirene Detectada” ou “Tudo normal”).

## Estrutura Modular

O projeto está dividido em cinco módulos principais:

- [`audio_capture.py`](https://github.com/lukebgds/Project_AudioSirenDetect-CNN/blob/main/src/audio_capture.py): Responsável por capturar segmentos de áudio com duração fixa, prontos para serem processados.  
- [`spectrogram_generator.py`](https://github.com/lukebgds/Project_AudioSirenDetect-CNN/blob/main/src/spectrogram_generator.py): Responsável pela geração e salvamento dos espectrogramas Mel a partir do áudio capturado.  
- [`image_preprocessor.py`](https://github.com/lukebgds/Project_AudioSirenDetect-CNN/blob/main/src/image_preprocessor.py): Realiza o pré-processamento das imagens dos espectrogramas para uso no modelo.  
- [`model_predictor.py`](https://github.com/lukebgds/Project_AudioSirenDetect-CNN/blob/main/src/model_predictor.py): Carrega o modelo de CNN treinado e realiza a predição das imagens.  
- [`main.py`](https://github.com/lukebgds/Project_AudioSirenDetect-CNN/blob/main/src/main.py): Orquestra o fluxo geral da aplicação, capturando áudio, gerando espectrogramas, realizando predições e exibindo o dashboard.

A execução do projeto é feita por meio do script [`run_AudioDetect.sh`](https://github.com/lukebgds/Project_AudioSirenDetect-CNN/blob/main/run_AudioDetect.sh), que ativa automaticamente o ambiente virtual, executa o arquivo `main.py` e encerra o ambiente após a finalização. Isso garante praticidade na inicialização e isolamento adequado do ambiente de dependências.

## Observação Final

Este projeto serve como base experimental e didática para estudos envolvendo reconhecimento de padrões sonoros com deep learning. Seu uso pode ser estendido para aplicações reais, com possíveis adaptações para outras categorias de sons ou melhorias no desempenho em tempo real.

---

### Integrantes do Grupo/Autores:

1. **Lucas Benício Gusmão da Silva**
2. **Rhyan Carlos da Silva Lima**
3. **Breno Almeida Custódio**
4. **Marcus Eduardo Dias Barbosa**
5. **Luis Gustavo de Albuquerque**
- **Turma:** 7MB
- **Horário:** MANHÂ
- **Curso:** Ciência da Computação

``Todos os Direitos Reservados``
