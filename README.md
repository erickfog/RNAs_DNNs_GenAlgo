# Classificação de Pneumonia em Raios-X com CNN

Este projeto implementa um sistema completo para classificação de pneumonia em imagens de raio-X usando Redes Neurais Convolucionais (CNNs) e o dataset PneumoniaMNIST.

## Estrutura do Projeto

```
├── README.md
├── requirements.txt
├── train.py
├── app.py
├── models/
│   └── pneumonia_cnn_model.keras
├── data/
│   └── .gitkeep
└── utils/
    └── data_loader.py
```

## Configuração do Ambiente

### 1. Criar e ativar ambiente virtual

```bash
# Usando venv (recomendado)
python -m venv pneumonia_env
source pneumonia_env/bin/activate  # Linux/Mac
# ou
pneumonia_env\Scripts\activate  # Windows

# Usando conda
conda create -n pneumonia_env python=3.9
conda activate pneumonia_env
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## Uso do Sistema

### 1. Treinamento do Modelo

Para treinar a CNN:

```bash
python train.py
```

O modelo treinado será salvo em `models/pneumonia_cnn_model.keras`.

### 2. Executar o App Streamlit

Para iniciar a aplicação web:

```bash
streamlit run app.py
```

Acesse http://localhost:8501 no seu navegador.

## Funcionalidades

- **Treinamento**: CNN otimizada para classificação binária (normal vs pneumonia)
- **Interface Web**: App Streamlit para upload e classificação de imagens
- **Visualização**: Gráficos de acurácia e perda durante o treinamento
- **Predição**: Classificação em tempo real com probabilidades

## Dataset

Utilizamos o **PneumoniaMNIST** do MedMNIST, que contém:
- 4,800 imagens de treino
- 1,200 imagens de validação  
- 1,200 imagens de teste
- Classes: Normal (0) e Pneumonia (1)

## Arquitetura da CNN

- Camadas convolucionais com ReLU
- MaxPooling para redução dimensional
- Dropout para regularização
- Camadas densas para classificação final
- Otimizador Adam com learning rate adaptativo

## Requisitos do Sistema

- Python 3.8+
- 8GB+ RAM (recomendado)
- GPU opcional (acelera treinamento)
- Navegador web moderno 