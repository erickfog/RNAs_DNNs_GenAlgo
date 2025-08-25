"""
Módulo para carregar e preparar o dataset PneumoniaMNIST
"""

import numpy as np
from medmnist import INFO, Evaluator
from medmnist import PneumoniaMNIST
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_pneumoniamnist():
    """
    Carrega o dataset PneumoniaMNIST e retorna os dados de treino, validação e teste
    
    Returns:
        tuple: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Carregar dataset
    data_train = PneumoniaMNIST(split='train', download=True)
    data_val = PneumoniaMNIST(split='val', download=True)
    data_test = PneumoniaMNIST(split='test', download=True)
    
    # Extrair dados
    X_train, y_train = data_train.imgs, data_train.labels
    X_val, y_val = data_val.imgs, data_val.labels
    X_test, y_test = data_test.imgs, data_test.labels
    
    # Converter para float32 e normalizar para [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Adicionar dimensão do canal (grayscale)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Converter labels para int32
    y_train = y_train.astype('int32').flatten()
    y_val = y_val.astype('int32').flatten()
    y_test = y_test.astype('int32').flatten()
    
    print(f"Dataset carregado:")
    print(f"Treino: {X_train.shape[0]} imagens")
    print(f"Validação: {X_val.shape[0]} imagens")
    print(f"Teste: {X_test.shape[0]} imagens")
    print(f"Classes: {np.unique(y_train)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Cria geradores de dados com data augmentation para treino
    
    Args:
        X_train: Imagens de treino
        y_train: Labels de treino
        X_val: Imagens de validação
        y_val: Labels de validação
        batch_size: Tamanho do batch
        
    Returns:
        tuple: (train_generator, val_generator)
    """
    # Data augmentation para treino
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Sem augmentation para validação
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    # Criar geradores
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator

def get_dataset_info():
    """
    Retorna informações sobre o dataset PneumoniaMNIST
    
    Returns:
        dict: Informações do dataset
    """
    try:
        info = INFO['pneumoniamnist']
        return {
            'name': 'PneumoniaMNIST',
            'description': info['description'],
            'num_classes': info['num_classes'],
            'class_names': ['Normal', 'Pneumonia'],
            'image_size': (28, 28),
            'channels': 1,
            'total_samples': 7200,
            'train_samples': 4800,
            'val_samples': 1200,
            'test_samples': 1200
        }
    except KeyError:
        # Fallback se a chave não existir
        return {
            'name': 'PneumoniaMNIST',
            'description': 'Dataset de raios-X de tórax para classificação de pneumonia',
            'num_classes': 2,
            'class_names': ['Normal', 'Pneumonia'],
            'image_size': (28, 28),
            'channels': 1,
            'total_samples': 7200,
            'train_samples': 4800,
            'val_samples': 1200,
            'test_samples': 1200
        } 