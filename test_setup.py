#!/usr/bin/env python3
"""
Script de teste para verificar a configuração do projeto
"""

import sys
import importlib
import os

def test_import(module_name, package_name=None):
    """Testa se um módulo pode ser importado"""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - FALHOU: {e}")
        return False

def test_tensorflow():
    """Testa funcionalidades do TensorFlow"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - OK")
        
        # Testar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU detectada: {len(gpus)} dispositivo(s)")
        else:
            print("💻 Usando CPU")
        
        # Testar operação básica
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        assert c.numpy().tolist() == [5, 7, 9]
        print("✅ Operações TensorFlow - OK")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow - FALHOU: {e}")
        return False

def test_medmnist():
    """Testa acesso ao MedMNIST"""
    try:
        from medmnist import INFO
        print("✅ MedMNIST - OK")
        
        # Verificar se PneumoniaMNIST está disponível
        if 'pneumoniamnist' in INFO:
            print("✅ Dataset PneumoniaMNIST disponível")
        else:
            print("⚠️  Dataset PneumoniaMNIST não encontrado")
        
        return True
    except Exception as e:
        print(f"❌ MedMNIST - FALHOU: {e}")
        return False

def test_streamlit():
    """Testa Streamlit"""
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} - OK")
        return True
    except Exception as e:
        print(f"❌ Streamlit - FALHOU: {e}")
        return False

def test_utils_module():
    """Testa módulo utils local"""
    try:
        from utils.data_loader import get_dataset_info
        info = get_dataset_info()
        print("✅ Módulo utils - OK")
        print(f"   Dataset: {info['name']}")
        return True
    except Exception as e:
        print(f"❌ Módulo utils - FALHOU: {e}")
        return False

def test_file_structure():
    """Testa estrutura de arquivos"""
    required_files = [
        "requirements.txt",
        "train.py",
        "app.py",
        "setup.py",
        "README.md"
    ]
    
    required_dirs = [
        "models",
        "data",
        "utils"
    ]
    
    print("\n📁 Verificando estrutura de arquivos...")
    
    all_files_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - NÃO ENCONTRADO")
            all_files_ok = False
    
    print("\n📂 Verificando diretórios...")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ - OK")
        else:
            print(f"❌ {directory}/ - NÃO ENCONTRADO")
            all_files_ok = False
    
    return all_files_ok

def main():
    """Função principal de teste"""
    print("🧪 Teste de Configuração do Projeto")
    print("=" * 40)
    
    tests = [
        ("TensorFlow", test_tensorflow),
        ("MedMNIST", test_medmnist),
        ("Streamlit", test_streamlit),
        ("Módulo Utils", test_utils_module),
        ("Estrutura de Arquivos", test_file_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testando {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 40)
    print("📊 RESUMO DOS TESTES")
    print("=" * 40)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! O projeto está configurado corretamente.")
        print("\n🚀 Você pode agora:")
        print("1. Treinar o modelo: python train.py")
        print("2. Executar o app: streamlit run app.py")
    else:
        print("⚠️  Alguns testes falharam. Verifique a configuração.")
        print("\n💡 Dicas:")
        print("- Verifique se o ambiente virtual está ativo")
        print("- Execute: pip install -r requirements.txt")
        print("- Verifique se todas as dependências foram instaladas")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 