"""
Script para treinar CNN de classificação de pneumonia em raios-X
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import warnings
warnings.filterwarnings('ignore')

# Importar módulo local
from utils.data_loader import load_pneumoniamnist, create_data_generators, get_dataset_info

def create_cnn_model(input_shape=(28, 28, 1), num_classes=2):
    """
    Cria uma CNN para classificação de pneumonia
    
    Args:
        input_shape: Formato das imagens de entrada
        num_classes: Número de classes (2: normal, pneumonia)
        
    Returns:
        model: Modelo Keras compilado
    """
    model = keras.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plota gráficos de acurácia e perda durante o treinamento
    
    Args:
        history: Histórico de treinamento do Keras
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de acurácia
    ax1.plot(history.history['accuracy'], label='Treino')
    ax1.plot(history.history['val_accuracy'], label='Validação')
    ax1.set_title('Acurácia do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico de perda
    ax2.plot(history.history['loss'], label='Treino')
    ax2.plot(history.history['val_loss'], label='Validação')
    ax2.set_title('Perda do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plota matriz de confusão
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        class_names: Nomes das classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.ylabel('Label Verdadeiro')
    plt.xlabel('Label Predito')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_with_thresholds(y_true, y_pred_proba, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Avalia o modelo com diferentes thresholds de confiança
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas
        thresholds: Lista de thresholds para testar
        
    Returns:
        dict: Resultados para cada threshold
    """
    results = {}
    
    for threshold in thresholds:
        # Aplicar threshold
        y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)
        
        # Calcular métricas
        cm = confusion_matrix(y_true, y_pred_threshold)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            accuracy = precision = recall = specificity = f1 = 0
        
        results[threshold] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    return results

def plot_threshold_analysis(threshold_results):
    """
    Plota análise de diferentes thresholds
    
    Args:
        threshold_results: Resultados para diferentes thresholds
    """
    thresholds = list(threshold_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [threshold_results[t][metric] for t in thresholds]
        axes[i].plot(thresholds, values, 'o-', linewidth=2, markersize=8)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_xlabel('Threshold')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].grid(True)
        axes[i].set_xlim(0.4, 1.0)
    
    # Remover subplot extra
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plota curva ROC
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Função principal para treinar o modelo
    """
    print("🚀 Iniciando treinamento da CNN para classificação de pneumonia...")
    
    # Configurar seed para reprodutibilidade
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Carregar dataset
    print("\n📊 Carregando dataset PneumoniaMNIST...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_pneumoniamnist()
    
    # Obter informações do dataset
    dataset_info = get_dataset_info()
    print(f"\n📋 Informações do Dataset:")
    for key, value in dataset_info.items():
        print(f"   {key}: {value}")
    
    # Criar modelo
    print("\n🏗️  Criando modelo CNN...")
    model = create_cnn_model()
    
    # Resumo do modelo
    print("\n📝 Resumo do Modelo:")
    model.summary()
    
    # Callbacks para melhor treinamento
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Treinar modelo
    print("\n🎯 Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Salvar modelo final
    print("\n💾 Salvando modelo...")
    os.makedirs('models', exist_ok=True)
    model.save('models/pneumonia_cnn_model.keras')
    print("✅ Modelo salvo em models/pneumonia_cnn_model.keras")
    
    # Plotar histórico de treinamento
    print("\n📈 Plotando gráficos de treinamento...")
    plot_training_history(history)
    
    # Avaliar modelo no conjunto de teste
    print("\n🧪 Avaliando modelo no conjunto de teste...")
    test_loss, test_accuracy, test_sparse_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Teste - Perda: {test_loss:.4f}")
    print(f"Teste - Acurácia: {test_accuracy:.4f}")
    
    # Predições no conjunto de teste
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Relatório de classificação
    print("\n📊 Relatório de Classificação:")
    class_names = dataset_info['class_names']
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plotar matriz de confusão
    print("\n🔍 Plotando matriz de confusão...")
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Análise com diferentes thresholds
    print("\n🎯 Analisando diferentes thresholds de confiança...")
    threshold_results = evaluate_with_thresholds(y_test, y_pred_proba)
    
    print("\n📊 Resultados por Threshold:")
    for threshold, metrics in threshold_results.items():
        print(f"\nThreshold {threshold:.1f}:")
        print(f"   Acurácia: {metrics['accuracy']:.4f}")
        print(f"   Precisão: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   Especificidade: {metrics['specificity']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
    
    # Plotar análise de thresholds
    print("\n📈 Plotando análise de thresholds...")
    plot_threshold_analysis(threshold_results)
    
    # Plotar curva ROC
    print("\n📊 Plotando curva ROC...")
    plot_roc_curve(y_test, y_pred_proba)
    
    print("\n🎉 Treinamento concluído com sucesso!")
    print("📁 Arquivos gerados:")
    print("   - models/pneumonia_cnn_model.keras (modelo treinado)")
    print("   - training_history.png (gráficos de treinamento)")
    print("   - confusion_matrix.png (matriz de confusão)")
    print("   - threshold_analysis.png (análise de thresholds)")
    print("   - roc_curve.png (curva ROC)")
    
    # Recomendação de threshold
    best_threshold = max(threshold_results.keys(), 
                        key=lambda t: threshold_results[t]['f1_score'])
    print(f"\n💡 **Recomendação:** Threshold de {best_threshold:.1f} para melhor F1-Score")

if __name__ == "__main__":
    main() 