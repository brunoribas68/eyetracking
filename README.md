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
- `--max-session-seconds 120` (para automaticamente após N segundos e salva tudo)
- `--training-display-target main|secondary|remote` (prepara estratégia de exibição para treino/UX)
  - em `secondary|remote`, abre a janela extra `Eye Tracking UX - Gaze Screen` com o ponto de olhar projetado
- `--gaze-overlay-mode cursor|heatmap_stub` (cursor atual ou modo base para heatmap)
- `--gaze-gain-x 2.2` e `--gaze-gain-y 2.0` (aumenta sensibilidade do ponto na tela projetada; útil quando fica preso no centro)

Pressione `q` para encerrar.

Também é possível interromper com `Ctrl+C` que a sessão parcial será salva.

## Pré-calibragem (novo)

Antes da coleta principal, o app roda uma etapa curta para melhorar a detecção de piscada:

1. Metade do tempo: mantenha os olhos abertos.
2. Metade do tempo: pisque naturalmente.

Com isso, o sistema calcula um `blink-ear-threshold` mais adequado para sua câmera/iluminação e também informa a cobertura de detecção de rosto/olhos.

## Precisão do ponto verde (melhoria)

O cursor verde agora usa a posição estimada da pupila/íris detectada no frame (não apenas o `gaze_x/gaze_y` normalizado), o que reduz casos em que o ponto “escapa” para bochecha/rosto.

No backend `opencv`, foi adicionado um filtro de qualidade para descartar caixas de olho improváveis (posição e proporção), reduzindo falsos positivos.

Se ainda ficar ruim no seu ambiente, rode com `--backend mediapipe` (quando disponível), que tende a ser mais robusto.

Exemplo com janela principal + tela de gaze separada:

```bash
python eyetracking_ux.py --output-dir runs/sessao_01 --show-window --training-display-target secondary --gaze-gain-x 2.4 --gaze-gain-y 2.2
```

## Saídas

### `session.json`

Resumo da sessão com:

- motivo de parada (`stop_reason`)
- backend utilizado
- quantidade de frames/fixações
- configuração de treino para próximos passos de UX (tela principal/secundária/remoto)

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
