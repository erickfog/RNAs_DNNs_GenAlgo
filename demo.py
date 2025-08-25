#!/usr/bin/env python3
"""
Script de demonstração rápida para testar o modelo treinado
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Adicionar utils ao path
sys.path.append('utils')
from data_loader import get_dataset_info

def create_demo_image(size=(28, 28), pattern='random'):
    """
    Cria uma imagem de demonstração para teste
    
    Args:
        size: Tamanho da imagem
        pattern: Padrão da imagem ('random', 'circle', 'lines')
    
    Returns:
        PIL Image: Imagem de demonstração
    """
    if pattern == 'random':
        # Imagem aleatória
        img_array = np.random.rand(*size) * 255
    elif pattern == 'circle':
        # Círculo no centro
        img_array = np.zeros(size)
        center = (size[0]//2, size[1]//2)
        radius = min(size) // 4
        
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img_array[mask] = 255
    elif pattern == 'lines':
        # Linhas horizontais
        img_array = np.zeros(size)
        for i in range(0, size[0], 4):
            img_array[i:i+2, :] = 255
    
    return Image.fromarray(img_array.astype('uint8'))

def load_and_test_model():
    """Carrega o modelo e testa com imagens de demonstração"""
    print("🚀 Demonstração do Sistema de Classificação de Pneumonia")
    print("=" * 60)
    
    # Verificar se o modelo existe
    model_path = "models/pneumonia_cnn_model.keras"
    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado!")
        print("   Execute primeiro: python train.py")
        return
    
    # Carregar modelo
    print("📥 Carregando modelo...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Informações do modelo
    print(f"\n📊 Informações do Modelo:")
    print(f"   Entrada: {model.input_shape}")
    print(f"   Saída: {model.output_shape}")
    print(f"   Parâmetros: {model.count_params():,}")
    
    # Informações do dataset
    dataset_info = get_dataset_info()
    class_names = dataset_info['class_names']
    
    # Testar com diferentes padrões
    patterns = ['random', 'circle', 'lines']
    
    fig, axes = plt.subplots(1, len(patterns), figsize=(15, 5))
    if len(patterns) == 1:
        axes = [axes]
    
    for i, pattern in enumerate(patterns):
        # Criar imagem de demonstração
        demo_img = create_demo_image(pattern=pattern)
        
        # Pré-processar para o modelo
        img_array = np.array(demo_img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=[0, -1])
        
        # Fazer predição
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Plotar resultado
        axes[i].imshow(demo_img, cmap='gray')
        axes[i].set_title(f"Padrão: {pattern}\n"
                         f"Classe: {class_names[predicted_class]}\n"
                         f"Confiança: {confidence:.3f}")
        axes[i].axis('off')
        
        print(f"\n🔍 Padrão '{pattern}':")
        print(f"   Classe predita: {class_names[predicted_class]}")
        print(f"   Confiança: {confidence:.3f}")
        print(f"   Probabilidades: {dict(zip(class_names, prediction))}")
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💾 Resultados salvos em 'demo_results.png'")
    
    # Testar com dataset real (se disponível)
    print("\n🧪 Testando com amostras do dataset...")
    try:
        from utils.data_loader import load_pneumoniamnist
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_pneumoniamnist()
        
        # Testar algumas imagens de teste
        num_samples = min(5, len(X_test))
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
        if num_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            # Imagem de teste
            test_img = X_test[idx]
            true_label = y_test[idx]
            
            # Fazer predição
            pred_input = np.expand_dims(test_img, axis=0)
            prediction = model.predict(pred_input, verbose=0)[0]
            pred_label = np.argmax(prediction)
            
            # Plotar
            axes[i].imshow(test_img.squeeze(), cmap='gray')
            axes[i].set_title(f"Verdadeiro: {class_names[true_label]}\n"
                             f"Predito: {class_names[pred_label]}\n"
                             f"Confiança: {np.max(prediction):.3f}")
            axes[i].axis('off')
            
            print(f"   Amostra {i+1}: {class_names[true_label]} → {class_names[pred_label]} "
                  f"(confiança: {np.max(prediction):.3f})")
        
        plt.tight_layout()
        plt.savefig('dataset_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Resultados do dataset salvos em 'dataset_test_results.png'")
        
    except Exception as e:
        print(f"⚠️  Não foi possível testar com dataset real: {e}")
    
    print("\n🎉 Demonstração concluída!")
    print("\n📋 Para usar o sistema completo:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    load_and_test_model() 