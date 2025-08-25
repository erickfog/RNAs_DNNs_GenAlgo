"""
App Streamlit para classificação de pneumonia em raios-X
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import get_dataset_info
import os

# Configurar TensorFlow para usar apenas CPU (evitar erros CUDA)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Configurações da página
st.set_page_config(
    page_title="Classificação de Pneumonia em Raios-X",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🫁 Classificação de Pneumonia em Raios-X")
st.markdown("---")

# Sidebar com informações e configurações
with st.sidebar:
    st.header("ℹ️ Sobre o Sistema")
    
    # Informações do dataset
    dataset_info = get_dataset_info()
    st.subheader("📊 Dataset")
    st.write(f"**Nome:** {dataset_info['name']}")
    st.write(f"**Descrição:** {dataset_info['description']}")
    st.write(f"**Tamanho das imagens:** {dataset_info['image_size']}")
    st.write(f"**Classes:** {', '.join(dataset_info['class_names'])}")
    
    st.subheader("🧠 Arquitetura CNN")
    st.write("""
    - **Camadas Convolucionais:** 3 blocos com filtros 32, 64, 128
    - **Regularização:** BatchNormalization + Dropout
    - **Pooling:** MaxPooling2D para redução dimensional
    - **Camadas Densas:** 512 → 256 → 2 neurônios
    - **Ativação:** Softmax para classificação
    """)
    
    st.subheader("⚙️ Configurações de Classificação")
    
    # Threshold configurável para classificação
    threshold = st.slider(
        "Threshold de Confiança (%)",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
        help="Confiança mínima para classificar como pneumonia. Valores mais altos = mais conservador."
    )
    
    # Explicação do threshold
    if threshold < 60:
        st.info("🔴 **Baixo Threshold:** Alta sensibilidade, pode gerar falsos positivos")
    elif threshold < 80:
        st.info("🟡 **Threshold Médio:** Equilibrado entre sensibilidade e especificidade")
    else:
        st.info("🟢 **Alto Threshold:** Alta especificidade, pode gerar falsos negativos")
    
    st.subheader("📈 Métricas")
    st.write("""
    - **Acurácia:** Medida de predições corretas
    - **Precisão:** Predições positivas corretas
    - **Recall:** Sensibilidade do modelo
    - **F1-Score:** Média harmônica entre precisão e recall
    """)

# Função para carregar modelo
@st.cache_resource
def load_model():
    """Carrega o modelo treinado"""
    try:
        model_path = "models/pneumonia_cnn_model.keras"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.error("❌ Modelo não encontrado! Execute primeiro o treinamento.")
            return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

# Função para pré-processar imagem
def preprocess_image(image):
    """Pré-processa imagem para o modelo"""
    # Converter para grayscale se necessário
    if image.mode != 'L':
        image = image.convert('L')
    
    # Redimensionar para 28x28
    image = image.resize((28, 28))
    
    # Converter para array e normalizar
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Adicionar dimensões necessárias
    img_array = np.expand_dims(img_array, axis=[0, -1])
    
    return img_array

# Função para fazer predição
def predict_pneumonia(model, image_array):
    """Faz predição usando o modelo"""
    try:
        prediction = model.predict(image_array, verbose=0)[0]
        return prediction  # Retornar o array completo, não apenas o primeiro elemento
    except Exception as e:
        st.error(f"❌ Erro na predição: {str(e)}")
        return None

# Função para classificar baseada no threshold
def classify_with_threshold(prediction, threshold_percent, class_names):
    """
    Classifica a imagem baseada no threshold de confiança
    
    Args:
        prediction: Array de probabilidades [normal, pneumonia]
        threshold_percent: Threshold em porcentagem
        class_names: Nomes das classes
        
    Returns:
        dict: Resultado da classificação
    """
    # Validar entrada
    if prediction is None or len(prediction) != 2:
        st.error("❌ Erro: Formato de predição inválido")
        return None
    
    threshold_decimal = threshold_percent / 100.0
    
    # Probabilidades
    prob_normal = float(prediction[0])
    prob_pneumonia = float(prediction[1])
    
    # Determinar classe baseada no threshold
    if prob_pneumonia >= threshold_decimal:
        predicted_class = 1  # Pneumonia
        confidence = prob_pneumonia
        classification_type = "CONFIANÇA ALTA"
    elif prob_pneumonia >= 0.5:
        predicted_class = 1  # Pneumonia (probabilidade > 50%)
        confidence = prob_pneumonia
        classification_type = "PROBABILIDADE"
    else:
        predicted_class = 0  # Normal
        confidence = prob_normal
        classification_type = "NORMAL"
    
    # Status da classificação
    if prob_pneumonia >= threshold_decimal:
        status = "PNEUMONIA DETECTADA"
        status_color = "danger"
    elif prob_pneumonia >= 0.5:
        status = "PNEUMONIA PROVÁVEL"
        status_color = "warning"
    else:
        status = "NORMAL"
        status_color = "success"
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'classification_type': classification_type,
        'status': status,
        'status_color': status_color,
        'probabilities': prediction,
        'threshold_used': threshold_decimal
    }

# Função para plotar resultado
def plot_prediction_result(image, prediction, class_names, classification_result):
    """Plota imagem e resultado da predição"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Imagem original
    ax1.imshow(image, cmap='gray')
    ax1.set_title("Imagem de Raios-X")
    ax1.axis('off')
    
    # Gráfico de barras com probabilidades
    bars = ax2.bar(class_names, prediction, color=['#2E8B57', '#DC143C'])
    ax2.set_title("Probabilidade de Classificação")
    ax2.set_ylabel("Probabilidade")
    ax2.set_ylim(0, 1)
    
    # Adicionar valores nas barras
    for bar, prob in zip(bars, prediction):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    # Adicionar linha do threshold
    threshold_line = classification_result['threshold_used']
    ax2.axhline(y=threshold_line, color='red', linestyle='--', 
                label=f'Threshold ({threshold_line:.2f})')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Carregar modelo
model = load_model()

# Interface principal
if model is not None:
    st.success("✅ Modelo carregado com sucesso!")
    
    # Upload de imagem
    st.header("📤 Upload de Imagem")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem de raio-X (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Carregar e exibir imagem
        image = Image.open(uploaded_file)
        
        # Layout em colunas
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Imagem Original")
            st.image(image, caption="Imagem carregada", use_container_width=True)
            
            # Informações da imagem
            st.write(f"**Formato:** {image.format}")
            st.write(f"**Tamanho:** {image.size}")
            st.write(f"**Modo:** {image.mode}")
        
        with col2:
            st.subheader("🔍 Análise")
            
            # Pré-processar imagem
            processed_image = preprocess_image(image)
            
            # Fazer predição
            if st.button("🚀 Classificar Imagem", type="primary"):
                with st.spinner("Analisando imagem..."):
                    prediction = predict_pneumonia(model, processed_image)
                    
                    if prediction is not None:
                        # Classificar com threshold
                        classification_result = classify_with_threshold(
                            prediction, threshold, dataset_info['class_names']
                        )
                        
                        # Verificar se a classificação foi bem-sucedida
                        if classification_result is None:
                            st.error("❌ Erro na classificação. Tente novamente.")
                        else:
                            # Exibir resultado principal
                            if classification_result['status_color'] == 'success':
                                st.success(f"✅ **{classification_result['status']}**")
                            elif classification_result['status_color'] == 'warning':
                                st.warning(f"⚠️ **{classification_result['status']}**")
                            else:
                                st.error(f"🚨 **{classification_result['status']}**")
                            
                            st.info(f"🎯 **Confiança:** {classification_result['confidence']:.3%}")
                            st.info(f"📊 **Tipo de Classificação:** {classification_result['classification_type']}")
                            
                            # Plotar resultado
                            fig = plot_prediction_result(image, prediction, dataset_info['class_names'], classification_result)
                            st.pyplot(fig)
                            
                            # Interpretação detalhada
                            st.subheader("📋 Interpretação Detalhada")
                            
                            if classification_result['predicted_class'] == 0:
                                st.info("💚 **Resultado:** A imagem foi classificada como **NORMAL**.")
                                st.write(f"   - Probabilidade de normal: {prediction[0]:.3%}")
                                st.write(f"   - Probabilidade de pneumonia: {prediction[1]:.3%}")
                                st.write(f"   - Threshold aplicado: {classification_result['threshold_used']:.1%}")
                            else:
                                st.warning("⚠️ **Resultado:** A imagem foi classificada como **PNEUMONIA**.")
                                st.write(f"   - Probabilidade de pneumonia: {prediction[1]:.3%}")
                                st.write(f"   - Probabilidade de normal: {prediction[0]:.3%}")
                                st.write(f"   - Threshold aplicado: {classification_result['threshold_used']:.1%}")
                                
                                if classification_result['classification_type'] == "CONFIANÇA ALTA":
                                    st.error("🚨 **Alta confiança:** Recomenda-se avaliação médica imediata.")
                                else:
                                    st.warning("⚠️ **Confiança moderada:** Recomenda-se avaliação médica profissional.")
                            
                            # Disclaimer médico
                            st.warning("""
                            ⚠️ **Aviso Médico:** 
                            Este sistema é apenas uma ferramenta educacional e de demonstração. 
                            **NÃO** deve ser usado para diagnóstico médico real. 
                            Sempre consulte um profissional de saúde qualificado.
                            """)
                    else:
                        st.error("❌ Erro na predição. Verifique se a imagem é válida.")

else:
    st.error("""
    ❌ **Modelo não disponível!**
    
    Para usar esta aplicação, você precisa primeiro treinar o modelo executando:
    
    ```bash
    python train.py
    ```
    
    Após o treinamento, o modelo será salvo em `models/pneumonia_cnn_model.keras`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🫁 Sistema de Classificação de Pneumonia em Raios-X usando CNN</p>
    <p>Desenvolvido com TensorFlow, Keras e Streamlit</p>
</div>
""", unsafe_allow_html=True) 