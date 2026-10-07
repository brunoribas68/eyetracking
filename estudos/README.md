# Estudos

Cada estudo (um estímulo / uma página testada) tem a sua pasta. Tudo o que é
preciso para gerar heatmap, scanpath e métricas por AOI fica junto:

```
estudos/<estudo>/
├── site.json           # (opcional) URL do site: o estimulo.png é capturado sozinho
├── estimulo.png        # print da página na resolução do teste (capturado ou colocado à mão)
├── estimulo.pagina.json# gerado na captura: URL, rolagem e caixas das AOIs por seletor
├── aois.json           # áreas de interesse: seletor CSS ou x/y/w/h em px do estímulo
├── navegador.json      # (opcional) login salvo para sites fechados; ignorado no git
├── sessoes/            # uma pasta por participante (gerada pela gravação, ignorada no git)
│   ├── P01/
│   └── P02/
└── resultados/         # gerada pela análise (ignorada no git)
    ├── heatmap/        # heatmap_<P>.png e group_heatmap.png
    ├── scanpath/       # scanpath_<P>.png
    ├── aois/           # aois.png (AOIs sobre o estímulo) e aoi_metrics.csv
    ├── summary.csv
    └── model_comparison.csv
```

## Fluxo

```bash
# 1. gravar cada participante dentro do estudo (vai para estudos/f1tv/sessoes/P01)
python eyetracking_ux.py --study estudos/f1tv --participant P01 \
    --task-text "..." --screen-width-cm 53 --distance-cm 60

# 2. analisar todas as sessões do estudo de uma vez
python analyze_session.py --study estudos/f1tv --compare-models
```

`--study` preenche `--stimulus`, `--aois`, `--out` e a lista de sessões; qualquer
um deles pode ser sobrescrito na linha de comando. Antes de rodar com
participantes, confira `resultados/aois/aois.png` para ver se as caixas caem
nos elementos certos.

## Usando um site em vez de imagem

Coloque um `site.json` no estudo:

```json
{
  "url": "https://f1tv.formula1.com",
  "viewport": "1920x1080",
  "wait_ms": 4000,
  "click": ["#onetrust-accept-btn-handler"],
  "hide": [".banner-promocional"]
}
```

- `click`: seletores clicados antes do print (ex.: aceitar cookies);
- `hide`: seletores escondidos (pop-ups, anúncios);
- `scroll`: `[x, y]` para capturar a página já rolada; `full_page: true` captura a página inteira.

Na primeira gravação com `--study`, se não houver `estimulo.png`, o site é
capturado **no tamanho da tela do teste** e todos os participantes seguintes
veem o mesmo print (o que permite somar os heatmaps). Para atualizar:
`--recapture`. Também dá para capturar antes, sem câmera:

```bash
python web_stimulus.py capture --study estudos/f1tv --viewport 1920x1080
```

Sites que exigem login (como o F1 TV): rode uma vez
`python web_stimulus.py login --study estudos/f1tv`, faça login na janela
que abrir e aperte ENTER; a sessão fica em `navegador.json` (não versione,
contém cookies) e é usada nas capturas.

### AOIs por seletor CSS

Com site, as AOIs podem apontar para elementos da página em vez de pixels;
a caixa é calculada na captura:

```json
[
  {"name": "busca",    "selector": "button[aria-label='Search']"},
  {"name": "destaque", "x": 150, "y": 300, "w": 400, "h": 180}
]
```

Para achar o seletor: botão direito no elemento → Inspecionar → botão direito
no HTML → Copiar → Copiar seletor. A captura avisa quais seletores não foram
encontrados ou ficaram fora da tela.

### Coordenadas (pensando no modo "ao vivo")

As caixas das AOIs ficam guardadas em coordenadas da **página** e o olhar é
medido na **vista** (tela); `PageView` em `web_stimulus.py` converte entre as
duas usando a rolagem. Hoje há uma vista só (rolagem fixa). Um modo em que o
participante navega de verdade só precisa registrar a sequência de vistas
(URL + rolagem ao longo do tempo) e agrupar a análise por URL.

## Criando um estudo novo

```bash
mkdir -p estudos/<nome>
# copie o print para estudos/<nome>/estimulo.png
# crie estudos/<nome>/aois.json: [{"name": "menu", "x": 20, "y": 80, "w": 160, "h": 220}, ...]
```
