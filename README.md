# Eye Tracking para UX (Webcam + MediaPipe)

Este projeto implementa um **pipeline único** para coletar dados oculares em tempo real usando webcam, com foco em métricas úteis para estudos de UX.

## O que o código entrega

- Captura da webcam em tempo real.
- Estimativa de ponto de olhar 2D normalizado (`gaze_x`, `gaze_y`) com landmarks de íris/olho do MediaPipe.
- Detecção simples de piscada via EAR (Eye Aspect Ratio).
- Detecção online de fixações por limiar espacial + duração mínima.
- Exportação de:
  - `frames.csv`: dados por frame.
  - `fixations.csv`: resumo das fixações detectadas.

> **Importante**: como todo eye tracking por webcam sem hardware dedicado, este método é aproximado e deve ser usado com calibração e validação para resultados científicos.

## Requisitos

- Python 3.10+
- Webcam

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Como rodar

```bash
python eyetracking_ux.py --output-dir runs/sessao_01 --show-window
```

Argumentos úteis:

- `--camera-index 0`: índice da câmera.
- `--fixation-threshold-px 60`: limiar espacial (pixels) para agrupar fixação.
- `--fixation-min-duration-ms 180`: duração mínima para validar fixação.
- `--blink-ear-threshold 0.21`: limiar EAR para considerar piscada.

Pressione `q` para encerrar.

## Formato do `frames.csv`

- `frame_idx`
- `timestamp_ms`
- `gaze_x`, `gaze_y` (0 a 1)
- `blink`
- `left_ear`, `right_ear`, `avg_ear`
- `fixation_id` (`-1` quando ainda não consolidada)

## Formato do `fixations.csv`

- `fixation_id`
- `start_ms`, `end_ms`, `duration_ms`
- `centroid_x`, `centroid_y` (pixels no frame)
- `samples`

## Próximos passos para TCC/UX

- Adicionar **calibração por pontos** (9-point calibration).
- Mapear AOIs e calcular métricas (TTFF, dwell time, revisits).
- Gerar heatmap e scanpath a partir de `frames.csv`/`fixations.csv`.
- Integrar protocolo experimental (tarefas, consentimento, logs de eventos).
