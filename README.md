# Eye Tracking Project

Projeto experimental de rastreamento ocular com webcam, usando técnicas de visão computacional/IA para captar informações do olhar em tempo real e apoiar estudos de UX.

## Objetivo
Construir uma base prática para obtenção de dados oculares (direção do olhar, piscada e landmarks faciais) que possa evoluir para análises de experiência do usuário.

## Scripts disponíveis
- `gaze-trackin2.py`: rastreamento de direção do olhar com `GazeTracking`.
- `gaze-tracking.py`: versão similar com marcação de pupilas e gravação simples em CSV.
- `landmark.py`: detecção de olhos por landmarks faciais com `dlib` (68 pontos).
- `google.py`: malha facial com `MediaPipe Face Mesh`.

## Tecnologias
- Python 3.8+
- OpenCV
- GazeTracking
- dlib + shape predictor 68 landmarks
- MediaPipe
- NumPy

## Como executar
1. Crie e ative um ambiente virtual.
2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute um dos scripts:
   ```bash
   python gaze-trackin2.py
   ```
   ou
   ```bash
   python landmark.py
   ```

Para sair, pressione `q` na janela do OpenCV.

## Observações
- O arquivo `shape_predictor_68_face_landmarks.dat` é necessário para `landmark.py`.
- Este repositório é uma base de prototipagem e ainda não implementa pipeline completo de avaliação UX (heatmap, AOI, métricas estatísticas e protocolo experimental).
