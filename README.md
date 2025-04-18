# Eye Tracking Project

Este é um projeto de rastreamento ocular que utiliza inteligência artificial para identificar e projetar em tempo real onde o usuário está olhando. Ele é útil para análise de comportamento, pesquisa de experiência do usuário, entre outras aplicações.

## 📋 Funcionalidades

- Captura de vídeo em tempo real usando a webcam.
- Rastreio preciso dos olhos com base na biblioteca `GazeTracking`.
- Identificação da direção do olhar (esquerda, direita, centro e piscada).
- Exibição anotada do quadro com a direção do olhar.
- Fácil personalização e extensibilidade para novas funcionalidades.

---

## 🛠 Tecnologias Utilizadas

- **Linguagem**: Python 3.8+
- **Bibliotecas Principais**:
  - [`OpenCV`](https://opencv.org/) - Para captura de vídeo e manipulação de imagens.
  - [`GazeTracking`](https://github.com/antoinelame/GazeTracking) - Para rastreamento ocular.
  - [`NumPy`](https://numpy.org/) - Para operações matemáticas e manipulação de arrays.
- **Ferramentas Adicionais** (opcional):
  - `Matplotlib` - Para visualização de dados.
  - `Seaborn` - Para análise de dados visuais.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos

Certifique-se de que você possui:
- Python instalado ([Download aqui](https://www.python.org/downloads/)).
- Ambiente virtual configurado (opcional, mas recomendado).

### Passo a Passo

1. Clone este repositório:
   ```
   git clone https://github.com/brunoribas68/eye-tracking-project.git
   cd eye-tracking-project
   
2. Ative o ambiente virtual:

     ```
      python -m venv .venv
      source .venv/bin/activate    # Linux/MacOS
      .venv\Scripts\activate       # Windows
     ```
3. Instale as dependências:

```
pip install -r requirements.txt
```

4. Execute o projeto:
```
python main.py
```

Para sair do programa, pressione 'q'.


   
## 🤝 Contribuições
### Sinta-se à vontade para abrir uma issue ou enviar um pull request com melhorias ou correções!


## ✨ Créditos
### Biblioteca de rastreamento ocular: GazeTracking.

### Inspirado em projetos de análise de comportamento humano.