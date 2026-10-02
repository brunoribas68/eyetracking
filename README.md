# Eye Tracking para UX com webcam

Protótipo (MVP) para coletar dados de rastreamento ocular com uma webcam comum
e transformá-los em métricas de UX (mapa de calor, scanpath, métricas por AOI).
Base do artigo *"Obtenção de dados de rastreamento ocular utilizando Inteligência
Artificial"* (Especialização em IA Aplicada — UFPR/SEPT).

## Como funciona

| Fase | O que acontece | Técnica |
|---|---|---|
| Detecção | rosto + 478 pontos faciais, incluindo íris | MediaPipe Face Mesh (CNN) · fallback Haar/Viola-Jones |
| Características | posição da íris no olho (x, y) + pose aproximada da cabeça | geometria dos landmarks |
| Piscada | Eye Aspect Ratio (EAR) com limiar calibrado por pessoa | Soukupová & Čech (2016) |
| Calibração | 9 alvos em tela cheia → regressão (linear ou polinomial 2º grau, ridge) | aprendizado supervisionado |
| Validação | 5 alvos **novos** → acurácia e precisão em px e graus | Holmqvist et al. (2012) |
| Suavização | filtro One Euro | Casiez et al. (2012) |
| Fixações | I-DT (dispersão) online | Salvucci & Goldberg (2000) |
| Análise | heatmap, scanpath, TTFF, dwell time, contagem por AOI | `analyze_session.py` |

## Instalação

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install "mediapipe==0.10.14"   # opcional, mas recomendado
```

> Use Python **64 bits** (3.9–3.12). O erro antigo de numpy/pandas no Windows vinha
> de Python 32 bits; o OpenCV já depende do numpy, então ele não tem como sair.

## Rodando uma sessão

```bash
python eyetracking_ux.py --participant P01 --output-dir runs/P01 \
    --stimulus estimulos/home.png --task-text "Encontre onde localizar uma loja" \
    --max-session-seconds 30 --screen-width-cm 53 --distance-cm 60 --show-camera
```

A janela abre em tela cheia. O participante aperta **ESPAÇO** em cada tela de
instrução; **Q/ESC** interrompe (o que já foi coletado é salvo).

- `--screen-width-cm` e `--distance-cm` permitem reportar erro em **graus visuais**
  (padrão da literatura). Meça a largura da área visível da tela e a distância olho–tela.
- `--calibration-model linear|poly2`, `--calibration-points 5|9|13`
- `--show-gaze` mostra o cursor do olhar sobre o estímulo (só demonstração: num teste
  real ele atrai o olhar e enviesa os dados).
- `python eyetracking_ux.py -h` lista todos os parâmetros.

Saídas em `runs/P01/`: `session.json` (configuração, calibração, métricas de
validação e de gravação), `frames.csv`, `fixations.csv`, `calibration_samples.csv`.

## Analisando

```bash
python analyze_session.py runs/P01 runs/P02 runs/P03 \
    --stimulus estimulos/home.png --aois aois_exemplo.json --out resultados --compare-models
```

Gera `group_heatmap.png`, `heatmap_<P>.png`, `scanpath_<P>.png`, `summary.csv`
(qualidade dos dados por participante), `aoi_metrics.csv` e `model_comparison.csv`
(linear × polinomial, com e sem pose da cabeça, treinado na calibração e avaliado
na validação).

## Testes

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Inclui um teste ponta a ponta com câmera, rosto e tela simulados.

## Limitações conhecidas

- Webcam comum fica tipicamente na casa de alguns graus de erro — bom para regiões
  da página (AOIs grandes), não para palavras ou ícones pequenos.
- A pose da cabeça é aproximada; movimentos grandes após a calibração degradam o
  mapeamento (recalibre ou use apoio de queixo).
- Óculos com reflexo, pouca luz e contraluz aumentam a perda de dados.
- O backend OpenCV não detecta piscadas (olho fechado vira "sem detecção").
