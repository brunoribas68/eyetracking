# Eye Tracking para UX (Webcam)

Pipeline para coletar dados oculares em tempo real via webcam com exportação em CSV para análises de UX.

## Compatibilidade (corrigido para seu caso)

Para evitar erros de instalação no Windows (especialmente Python 32-bit/x86), o projeto agora depende **somente de OpenCV** no `requirements.txt`.

- Backend padrão: `opencv` (compatível)
- Backend opcional: `mediapipe` (instalação manual, se disponível)

## Requisitos

- Python 3.10+
- Webcam

## Instalação

```bash
pip install -r requirements.txt
```

### MediaPipe opcional

Se quiser tentar maior precisão e seu ambiente suportar:

```bash
pip install mediapipe
```

## Como rodar

```bash
python eyetracking_ux.py --output-dir runs/sessao_01 --show-window
```

Parâmetros úteis:

- `--backend auto|mediapipe|opencv` (padrão: `auto`)
- `--camera-index 0`
- `--fixation-threshold-px 60`
- `--fixation-min-duration-ms 180`
- `--blink-ear-threshold 0.21`
- `--precheck-seconds 8` (etapa antes da calibragem para validar detecção e ajustar EAR)
- `--skip-precheck` (pula essa etapa)

Pressione `q` para encerrar.

## Pré-calibragem (novo)

Antes da coleta principal, o app roda uma etapa curta para melhorar a detecção de piscada:

1. Metade do tempo: mantenha os olhos abertos.
2. Metade do tempo: pisque naturalmente.

Com isso, o sistema calcula um `blink-ear-threshold` mais adequado para sua câmera/iluminação e também informa a cobertura de detecção de rosto/olhos.

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

## Dica para seu erro atual

Se aparecer erro de `numpy`/`pandas`, apague a `.venv` e recrie:

```bash
# PowerShell
Deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\eyetracking_ux.py --backend opencv --show-window
```
