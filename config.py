"""
Arquivo de configuração para o projeto de classificação de pneumonia
"""

# Configurações do Dataset
DATASET_CONFIG = {
    'name': 'PneumoniaMNIST',
    'image_size': (28, 28),
    'channels': 1,
    'num_classes': 2,
    'class_names': ['Normal', 'Pneumonia'],
    'train_samples': 4800,
    'val_samples': 1200,
    'test_samples': 1200
}

# Configurações do Modelo
MODEL_CONFIG = {
    'input_shape': (28, 28, 1),
    'num_classes': 2,
    'filters': [32, 64, 128],
    'kernel_size': (3, 3),
    'pool_size': (2, 2),
    'dropout_rates': [0.25, 0.25, 0.25, 0.5, 0.3],
    'dense_layers': [512, 256]
}

# Configurações de Treinamento
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'min_lr': 1e-7
}

# Configurações de Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 10,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Configurações do App Streamlit
STREAMLIT_CONFIG = {
    'page_title': 'Classificação de Pneumonia em Raios-X',
    'page_icon': '🫁',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Configurações de Salvamento
SAVE_CONFIG = {
    'model_dir': 'models',
    'model_filename': 'pneumonia_cnn_model.keras',
    'best_model_filename': 'best_model.keras',
    'history_plot_filename': 'training_history.png',
    'confusion_matrix_filename': 'confusion_matrix.png'
}

# Configurações de Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'training.log'
}

# Configurações de Avaliação
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1-score'],
    'confusion_matrix': True,
    'classification_report': True,
    'roc_curve': False  # Desabilitado para dataset binário pequeno
} 