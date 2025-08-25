#!/usr/bin/env python3
"""
Script de instalação inteligente para o projeto de classificação de pneumonia
"""

import os
import sys
import subprocess
import platform
import shutil

def check_python_version():
    """Verifica a versão do Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        return False
    
    print("✅ Versão do Python compatível!")
    return True

def check_system_info():
    """Verifica informações do sistema"""
    print(f"💻 Sistema: {platform.system()} {platform.release()}")
    print(f"🏗️  Arquitetura: {platform.machine()}")
    
    # Verificar memória disponível
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 RAM: {memory.total // (1024**3):.1f} GB total, {memory.available // (1024**3):.1f} GB disponível")
        
        if memory.total < 4 * (1024**3):  # Menos de 4GB
            print("⚠️  RAM baixa detectada. Use requirements-minimal.txt")
            return "minimal"
        elif memory.total < 8 * (1024**3):  # Menos de 8GB
            print("⚠️  RAM moderada. Use requirements-prod.txt")
            return "production"
        else:
            print("✅ RAM suficiente para instalação completa")
            return "full"
    except ImportError:
        print("ℹ️  psutil não disponível. Assumindo RAM suficiente.")
        return "full"

def check_gpu():
    """Verifica se há GPU disponível"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU detectada: {len(gpus)} dispositivo(s)")
            return True
        else:
            print("💻 Nenhuma GPU detectada - usando CPU")
            return False
    except ImportError:
        print("ℹ️  TensorFlow não instalado ainda")
        return False

def select_requirements_file(system_type, has_gpu):
    """Seleciona o arquivo de requirements apropriado"""
    if system_type == "minimal":
        return "requirements-minimal.txt"
    elif system_type == "production":
        return "requirements-prod.txt"
    else:
        if has_gpu:
            return "requirements.txt"  # Versão completa com suporte GPU
        else:
            return "requirements-prod.txt"  # Versão estável para CPU

def install_requirements(requirements_file):
    """Instala as dependências do arquivo selecionado"""
    print(f"\n📦 Instalando dependências de {requirements_file}...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], check=True, capture_output=True, text=True)
        
        print("✅ Dependências instaladas com sucesso!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro na instalação: {e}")
        print(f"Saída de erro: {e.stderr}")
        return False

def create_directories():
    """Cria diretórios necessários"""
    directories = ["models", "data", "utils"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Diretório '{directory}' criado")
        else:
            print(f"✅ Diretório '{directory}' já existe")

def test_installation():
    """Testa se a instalação foi bem-sucedida"""
    print("\n🧪 Testando instalação...")
    
    try:
        # Testar imports básicos
        import numpy as np
        print("✅ NumPy - OK")
        
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} - OK")
        
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} - OK")
        
        from medmnist import INFO
        print("✅ MedMNIST - OK")
        
        print("🎉 Todos os módulos principais funcionando!")
        return True
        
    except ImportError as e:
        print(f"❌ Erro de import: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 Instalação Inteligente do Projeto de Classificação de Pneumonia")
    print("=" * 70)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Verificar sistema
    system_type = check_system_info()
    has_gpu = check_gpu()
    
    # Selecionar requirements
    requirements_file = select_requirements_file(system_type, has_gpu)
    print(f"\n📋 Arquivo selecionado: {requirements_file}")
    
    # Criar diretórios
    create_directories()
    
    # Instalar dependências
    if not install_requirements(requirements_file):
        print("❌ Falha na instalação das dependências!")
        sys.exit(1)
    
    # Testar instalação
    if not test_installation():
        print("❌ Falha no teste da instalação!")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🎉 Instalação concluída com sucesso!")
    print("\n📋 Próximos passos:")
    print("1. Treinar o modelo: python train.py")
    print("2. Executar o app: streamlit run app.py")
    print("3. Testar o sistema: python quick_test.py")
    
    print(f"\n💡 Dica: Use '{requirements_file}' para reinstalar dependências específicas")

if __name__ == "__main__":
    main()
