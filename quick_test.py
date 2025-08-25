#!/usr/bin/env python3
"""
Script de teste rápido para verificar o sistema
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os

# Adicionar utils ao path
sys.path.append('utils')

def test_imports():
    """Testa se todos os módulos podem ser importados"""
    print("🧪 Testando imports...")
    
    try:
        from utils.data_loader import get_dataset_info
        print("✅ utils.data_loader - OK")
    except Exception as e:
        print(f"❌ utils.data_loader - FALHOU: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit - OK")
    except Exception as e:
        print(f"❌ streamlit - FALHOU: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ tensorflow {tf.__version__} - OK")
    except Exception as e:
        print(f"❌ tensorflow - FALHOU: {e}")
        return False
    
    return True

def test_dataset_info():
    """Testa se conseguimos obter informações do dataset"""
    print("\n📊 Testando informações do dataset...")
    
    try:
        from utils.data_loader import get_dataset_info
        info = get_dataset_info()
        print(f"✅ Dataset: {info['name']}")
        print(f"✅ Classes: {info['class_names']}")
        print(f"✅ Tamanho: {info['image_size']}")
        return True
    except Exception as e:
        print(f"❌ Erro ao obter informações do dataset: {e}")
        return False

def test_model_loading():
    """Testa se conseguimos carregar o modelo (se existir)"""
    print("\n🤖 Testando carregamento do modelo...")
    
    model_path = "models/pneumonia_cnn_model.keras"
    
    if not os.path.exists(model_path):
        print("⚠️  Modelo não encontrado - isso é normal se ainda não foi treinado")
        return True
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo carregado: {model.input_shape} → {model.output_shape}")
        print(f"✅ Parâmetros: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

def test_prediction_pipeline():
    """Testa o pipeline de predição"""
    print("\n🔮 Testando pipeline de predição...")
    
    try:
        # Criar imagem de teste
        test_image = np.random.rand(28, 28) * 255
        test_image = Image.fromarray(test_image.astype('uint8'))
        
        # Pré-processar
        img_array = np.array(test_image).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=[0, -1])
        
        print(f"✅ Imagem de teste criada: {img_array.shape}")
        
        # Se o modelo existir, testar predição
        model_path = "models/pneumonia_cnn_model.keras"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            prediction = model.predict(img_array, verbose=0)[0]
            print(f"✅ Predição realizada: {prediction}")
            print(f"✅ Classe predita: {'Pneumonia' if prediction[1] > 0.5 else 'Normal'}")
        else:
            print("⚠️  Modelo não encontrado - pulando teste de predição")
        
        return True
    except Exception as e:
        print(f"❌ Erro no pipeline de predição: {e}")
        return False

def test_threshold_classification():
    """Testa a classificação baseada em threshold"""
    print("\n🎯 Testando classificação com threshold...")
    
    try:
        # Simular predições
        predictions = [
            [0.8, 0.2],  # Normal com alta confiança
            [0.3, 0.7],  # Pneumonia com alta confiança
            [0.6, 0.4],  # Normal com confiança moderada
            [0.45, 0.55] # Pneumonia com confiança baixa
        ]
        
        thresholds = [0.5, 0.6, 0.7]
        
        for threshold in thresholds:
            print(f"\nThreshold {threshold:.1f}:")
            for i, pred in enumerate(predictions):
                prob_pneumonia = pred[1]
                if prob_pneumonia >= threshold:
                    status = "PNEUMONIA"
                elif prob_pneumonia >= 0.5:
                    status = "PNEUMONIA PROVÁVEL"
                else:
                    status = "NORMAL"
                
                print(f"   Predição {i+1}: {status} (conf: {prob_pneumonia:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Erro na classificação com threshold: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🚀 Teste Rápido do Sistema de Classificação de Pneumonia")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Informações do Dataset", test_dataset_info),
        ("Carregamento do Modelo", test_model_loading),
        ("Pipeline de Predição", test_prediction_pipeline),
        ("Classificação com Threshold", test_threshold_classification)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! O sistema está funcionando corretamente.")
        print("\n🚀 Próximos passos:")
        if not os.path.exists("models/pneumonia_cnn_model.keras"):
            print("1. Treinar o modelo: python train.py")
        print("2. Executar o app: streamlit run app.py")
    else:
        print("⚠️  Alguns testes falharam. Verifique a configuração.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 