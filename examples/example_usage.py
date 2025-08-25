"""
Exemplos de uso do sistema de classificação de pneumonia
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Adicionar utils ao path
sys.path.append('../utils')
from data_loader import load_pneumoniamnist, get_dataset_info

def example_1_basic_prediction():
    """Exemplo 1: Predição básica com uma imagem"""
    print("🔍 Exemplo 1: Predição Básica")
    print("-" * 40)
    
    # Carregar modelo (se existir)
    model_path = "../models/pneumonia_cnn_model.keras"
    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado! Execute primeiro o treinamento.")
        return
    
    model = tf.keras.models.load_model(model_path)
    
    # Criar imagem de exemplo
    img_array = np.random.rand(28, 28) * 255
    img = Image.fromarray(img_array.astype('uint8'))
    
    # Pré-processar
    processed = img_array.astype('float32') / 255.0
    processed = np.expand_dims(processed, axis=[0, -1])
    
    # Predição
    prediction = model.predict(processed, verbose=0)[0]
    class_names = ['Normal', 'Pneumonia']
    predicted_class = np.argmax(prediction)
    
    print(f"Imagem: {img.size}")
    print(f"Classe predita: {class_names[predicted_class]}")
    print(f"Confiança: {np.max(prediction):.3f}")
    print(f"Probabilidades: {dict(zip(class_names, prediction))}")

def example_2_batch_prediction():
    """Exemplo 2: Predição em lote"""
    print("\n🔍 Exemplo 2: Predição em Lote")
    print("-" * 40)
    
    try:
        # Carregar dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_pneumoniamnist()
        
        # Selecionar algumas imagens de teste
        num_samples = 10
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        sample_images = X_test[indices]
        sample_labels = y_test[indices]
        
        # Carregar modelo
        model_path = "../models/pneumonia_cnn_model.keras"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            
            # Predições em lote
            predictions = model.predict(sample_images, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Calcular acurácia
            accuracy = np.mean(predicted_labels == sample_labels)
            print(f"Acurácia no lote de {num_samples} imagens: {accuracy:.3f}")
            
            # Mostrar algumas predições
            for i in range(min(5, num_samples)):
                true_label = "Normal" if sample_labels[i] == 0 else "Pneumonia"
                pred_label = "Normal" if predicted_labels[i] == 0 else "Pneumonia"
                confidence = np.max(predictions[i])
                print(f"  Imagem {i+1}: {true_label} → {pred_label} (conf: {confidence:.3f})")
        else:
            print("❌ Modelo não encontrado!")
            
    except Exception as e:
        print(f"❌ Erro: {e}")

def example_3_data_visualization():
    """Exemplo 3: Visualização dos dados"""
    print("\n🔍 Exemplo 3: Visualização dos Dados")
    print("-" * 40)
    
    try:
        # Carregar dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_pneumoniamnist()
        
        # Estatísticas básicas
        print(f"Treino: {X_train.shape[0]} imagens")
        print(f"Validação: {X_val.shape[0]} imagens")
        print(f"Teste: {X_test.shape[0]} imagens")
        
        # Distribuição de classes
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        print(f"\nDistribuição de classes (treino):")
        class_names = ['Normal', 'Pneumonia']
        for i, (cls, count) in enumerate(zip(train_classes, train_counts)):
            print(f"  {class_names[cls]}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Visualizar algumas imagens
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(8):
            if i < len(X_train):
                axes[i].imshow(X_train[i].squeeze(), cmap='gray')
                axes[i].set_title(f"{class_names[y_train[i]]}")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('../example_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualização salva em 'example_visualization.png'")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

def example_4_model_evaluation():
    """Exemplo 4: Avaliação completa do modelo"""
    print("\n🔍 Exemplo 4: Avaliação do Modelo")
    print("-" * 40)
    
    try:
        # Carregar modelo
        model_path = "../models/pneumonia_cnn_model.keras"
        if not os.path.exists(model_path):
            print("❌ Modelo não encontrado!")
            return
        
        model = tf.keras.models.load_model(model_path)
        
        # Carregar dados de teste
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_pneumoniamnist()
        
        # Avaliar modelo
        test_loss, test_accuracy, test_sparse_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Perda no teste: {test_loss:.4f}")
        print(f"Acurácia no teste: {test_accuracy:.4f}")
        
        # Predições
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Matriz de confusão
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_test, y_pred)
        class_names = ['Normal', 'Pneumonia']
        
        print(f"\nMatriz de Confusão:")
        print(cm)
        
        # Relatório de classificação
        print(f"\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
    except Exception as e:
        print(f"❌ Erro: {e}")

def main():
    """Executa todos os exemplos"""
    print("🚀 Exemplos de Uso do Sistema de Classificação de Pneumonia")
    print("=" * 70)
    
    examples = [
        example_1_basic_prediction,
        example_2_batch_prediction,
        example_3_data_visualization,
        example_4_model_evaluation
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Erro no exemplo: {e}")
        
        print("\n" + "=" * 70)
    
    print("🎉 Todos os exemplos foram executados!")
    print("\n💡 Para mais informações, consulte:")
    print("   - README.md")
    print("   - train.py")
    print("   - app.py")

if __name__ == "__main__":
    main() 