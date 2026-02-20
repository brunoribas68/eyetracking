# Eye Tracking para UX (Webcam)

Pipeline para coletar dados oculares em tempo real via webcam com exportação em CSV para análises de UX.

## Novidade importante (compatibilidade)

O projeto agora suporta **2 backends**:

- `mediapipe` (mais preciso quando disponível)
- `opencv` (fallback sem MediaPipe, mais compatível)

No modo padrão (`--backend auto`), o script tenta MediaPipe e, se não estiver instalado/disponível no Python da máquina, cai automaticamente para OpenCV.

## Requisitos

- Python 3.10+
- Webcam

Instalação base (sempre funciona para backend OpenCV):

```bash
pip install -r requirements.txt
```

### MediaPipe (opcional)

Se quiser usar backend `mediapipe`, instale manualmente quando houver wheel para sua versão de Python/SO:

```bash
pip install mediapipe
```

> Em algumas combinações (ex.: Python muito novo), pode não haver pacote disponível.

## Como rodar

```bash
python eyetracking_ux.py --output-dir runs/sessao_01 --show-window
```

Parâmetros úteis:

- `--backend auto|mediapipe|opencv`
- `--camera-index 0`
- `--fixation-threshold-px 60`
- `--fixation-min-duration-ms 180`
- `--blink-ear-threshold 0.21`

Pressione `q` para encerrar.

## Saídas

### `frames.csv`

- `frame_idx`
- `timestamp_ms`
- `gaze_x`, `gaze_y` (0..1)
- `blink`
- `left_ear`, `right_ear`, `avg_ear`
- `fixation_id`
- `backend`

### `fixations.csv`

- `fixation_id`
- `start_ms`, `end_ms`, `duration_ms`
- `centroid_x`, `centroid_y`
- `samples`

## Observação científica

Eye tracking por webcam é aproximação. Para TCC/publicação, inclua calibração, protocolo experimental e validação de erro.
