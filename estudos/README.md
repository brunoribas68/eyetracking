# Estudos

Cada estudo (um estímulo / uma página testada) tem a sua pasta. Tudo o que é
preciso para gerar heatmap, scanpath e métricas por AOI fica junto:

```
estudos/<estudo>/
├── estimulo.png        # print da página na resolução usada no teste (você coloca)
├── aois.json           # áreas de interesse, em px do estimulo.png
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
# 1. gravar cada participante dentro do estudo
python eyetracking_ux.py --participant P01 --output-dir estudos/f1tv/sessoes/P01 \
    --stimulus estudos/f1tv/estimulo.png --task-text "..." --screen-width-cm 53 --distance-cm 60

# 2. analisar todas as sessões do estudo de uma vez
python analyze_session.py --study estudos/f1tv --compare-models
```

`--study` preenche `--stimulus`, `--aois`, `--out` e a lista de sessões; qualquer
um deles pode ser sobrescrito na linha de comando. Antes de rodar com
participantes, confira `resultados/aois/aois.png` para ver se as caixas caem
nos elementos certos.

## Criando um estudo novo

```bash
mkdir -p estudos/<nome>
# copie o print para estudos/<nome>/estimulo.png
# crie estudos/<nome>/aois.json: [{"name": "menu", "x": 20, "y": 80, "w": 160, "h": 220}, ...]
```
