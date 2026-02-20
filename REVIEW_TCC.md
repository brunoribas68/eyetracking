# Revisão do projeto para o TCC

## Qual era a ideia do projeto
A ideia central deste repositório é **obter dados de rastreamento ocular em tempo real pela webcam**, usando IA/visão computacional, para apoiar análise de **Experiência do Usuário (UX)**.

Na prática, o projeto implementa um **protótipo experimental** com três trilhas:

1. **Rastreamento de direção do olhar com GazeTracking** (`gaze-trackin2.py` e `gaze-tracking.py`).
2. **Detecção de landmarks faciais com dlib + shape predictor 68 pontos** (`landmark.py`).
3. **Malha facial com MediaPipe** (`google.py`).

Isso está alinhado ao seu texto do TCC: captura ocular, detecção de olhos/face, e uso como fonte de dados para estudos de UX.

## O que o projeto já entrega
- Captura de vídeo em tempo real via webcam.
- Detecção básica de estado do olhar (esquerda/direita/centro/piscada) com GazeTracking.
- Marcação de olhos/landmarks para validação visual da detecção.

## Limitações atuais (importantes para banca)
- Não há pipeline completo de experimento UX (tarefas, AOIs, métricas, relatório).
- Não há tratamento robusto de ruído/data-loss (saccades, piscadas, frames inválidos, calibração).
- Não há avaliação quantitativa formal (acurácia, precisão, recall, erro angular).
- Estrutura está em scripts soltos; falta consolidação para reprodutibilidade científica.

## Como posicionar no TCC
Você pode apresentar este repositório como:
- **Prova de conceito funcional** para coleta inicial de dados oculares em ambiente controlado.
- **Base de experimentação** para evoluir para métricas de UX (fixação, tempo até primeira fixação, mapas de calor, scanpath).

## Próximos passos recomendados
1. Criar um script único de execução e padronizar dataset de saída (`csv/json`) por frame.
2. Incluir calibração simples e filtro temporal (suavização) para reduzir ruído.
3. Definir protocolo UX (tarefas, participantes, tempo, critérios) e métricas.
4. Gerar visualizações (heatmap/scanpath) para análise comparativa.
5. Documentar resultados com limitações e ameaças à validade.
