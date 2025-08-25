# Classificação de Pneumonia em Raios-X com CNN

Este projeto implementa um sistema completo para classificação de pneumonia em imagens de raio-X usando Redes Neurais Convolucionais (CNNs) e o dataset PneumoniaMNIST.

## 🚀 **Instalação Rápida**

### **Opção 1: Instalação Completa (Recomendado para desenvolvimento)**
```bash
pip install -r requirements.txt
```

### **Opção 2: Instalação para Produção (Versões estáveis)**
```bash
pip install -r requirements-prod.txt
```

### **Opção 3: Instalação Minimal (Apenas essencial)**
```bash
pip install -r requirements-minimal.txt
```

### **Opção 4: Deploy (Apenas para executar o app)**
```bash
pip install -r requirements-deploy.txt
```

## 🚀 **Deploy em Produção**

### **Para Render, Heroku, Railway, etc.**
```bash
# Use o requirements-deploy.txt (muito mais leve)
pip install -r requirements-deploy.txt

# Configure o ambiente
python deploy-config.py

# Execute o app
streamlit run app.py
```

### **Vantagens do Modo Deploy**
- ✅ **Muito mais leve**: Sem TensorFlow, PyTorch, CUDA
- ✅ **Deploy rápido**: Instalação em segundos
- ✅ **Funcionalidade básica**: App funciona para demonstração
- ✅ **Sem erros CUDA**: Funciona em qualquer ambiente
- ⚠️ **Limitações**: Predições simuladas (não reais)

### **Para Funcionalidade Completa em Deploy**
```bash
# Instalar TensorFlow (opcional)
pip install tensorflow

# Treinar modelo localmente e fazer upload
python train.py
# Upload do arquivo models/pneumonia_cnn_model.keras
```

## 📋 **Requisitos do Sistema**

- **Python**: 3.8 - 3.11 (recomendado 3.9)
- **RAM**: 4GB+ (8GB+ recomendado)
- **GPU**: Opcional (CPU funciona bem)
- **Sistema**: Linux, macOS, Windows

## 🛠️ **Configuração do Ambiente**

### **1. Criar e ativar ambiente virtual**

```bash
# Usando venv (recomendado)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Usando conda
conda create -n pneumonia_env python=3.9
conda activate pneumonia_env
```

### **2. Instalar dependências**

```bash
# Para desenvolvimento
pip install -r requirements.txt

# Para produção
pip install -r requirements-prod.txt

# Para casos simples
pip install -r requirements-minimal.txt
```

## 🎯 **Uso do Sistema**

### **1. Treinamento do Modelo**

```bash
python train.py
```

O modelo treinado será salvo em `models/pneumonia_cnn_model.keras`.

### **2. Executar o App Streamlit**

```bash
streamlit run app.py
```

Acesse http://localhost:8501 no seu navegador.

### **3. Teste Rápido**

```bash
python quick_test.py
```

## 📁 **Estrutura do Projeto**

```
├── README.md
├── requirements.txt          # Desenvolvimento
├── requirements-prod.txt     # Produção
├── requirements-minimal.txt  # Minimal
├── train.py                 # Treinamento
├── app.py                   # App Streamlit
├── quick_test.py            # Teste rápido
├── models/                  # Modelos treinados
├── data/                    # Datasets
└── utils/                   # Utilitários
```

## 🔧 **Solução de Problemas**

### **Erro CUDA/GPU**
Se encontrar erros relacionados ao CUDA, o sistema já está configurado para usar CPU automaticamente.

### **Versões Incompatíveis**
Use `requirements-prod.txt` para versões estáveis ou `requirements-minimal.txt` para dependências básicas.

### **Problemas de Memória**
- Reduza o batch size no `train.py`
- Use `requirements-minimal.txt`
- Feche outros aplicativos

## 📊 **Funcionalidades**

- **Treinamento**: CNN otimizada para classificação binária
- **Interface Web**: App Streamlit para upload e classificação
- **Threshold Configurável**: Ajuste a sensibilidade do modelo
- **Visualização**: Gráficos de acurácia, perda e ROC
- **Classificação Inteligente**: Baseada em confiança, não apenas classe

## 🧠 **Arquitetura da CNN**

- **Camadas Convolucionais**: 3 blocos (32, 64, 128 filtros)
- **Regularização**: BatchNormalization + Dropout
- **Pooling**: MaxPooling2D para redução dimensional
- **Camadas Densas**: 512 → 256 → 2 neurônios
- **Ativação**: Softmax para classificação

## 📈 **Métricas de Avaliação**

- **Acurácia**: Predições corretas
- **Precisão**: Predições positivas corretas
- **Recall**: Sensibilidade do modelo
- **F1-Score**: Média harmônica
- **Curva ROC**: Capacidade discriminativa
- **Análise de Threshold**: Otimização de sensibilidade

## ⚠️ **Aviso Médico**

Este sistema é **APENAS** uma ferramenta educacional e de demonstração. **NÃO** deve ser usado para diagnóstico médico real. Sempre consulte um profissional de saúde qualificado.

## 🤝 **Contribuição**

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 **Licença**

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes. 