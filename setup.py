#!/usr/bin/env python3
"""
Script de setup para configurar o ambiente do projeto
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Executa um comando e exibe o resultado"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} concluído com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro em {description}: {e}")
        print(f"Saída de erro: {e.stderr}")
        return False

def check_python_version():
    """Verifica a versão do Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário!")
        return False
    
    print("✅ Versão do Python compatível!")
    return True

def create_virtual_environment():
    """Cria ambiente virtual"""
    if os.path.exists("pneumonia_env"):
        print("✅ Ambiente virtual já existe!")
        return True
    
    print("🔄 Criando ambiente virtual...")
    
    if platform.system() == "Windows":
        command = "python -m venv pneumonia_env"
    else:
        command = "python3 -m venv pneumonia_env"
    
    return run_command(command, "Criação do ambiente virtual")

def activate_virtual_environment():
    """Ativa o ambiente virtual"""
    if platform.system() == "Windows":
        activate_script = "pneumonia_env\\Scripts\\activate"
    else:
        activate_script = "source pneumonia_env/bin/activate"
    
    print(f"\n🔧 Para ativar o ambiente virtual, execute:")
    print(f"   {activate_script}")
    
    if platform.system() != "Windows":
        print("\n🔧 Ou use:")
        print("   source pneumonia_env/bin/activate")

def install_requirements():
    """Instala as dependências"""
    print("\n📦 Instalando dependências...")
    
    # Verificar se o ambiente virtual está ativo
    if "VIRTUAL_ENV" not in os.environ:
        print("⚠️  Ambiente virtual não está ativo!")
        print("   Ative o ambiente virtual primeiro:")
        if platform.system() == "Windows":
            print("   pneumonia_env\\Scripts\\activate")
        else:
            print("   source pneumonia_env/bin/activate")
        return False
    
    return run_command("pip install -r requirements.txt", "Instalação das dependências")

def create_directories():
    """Cria diretórios necessários"""
    directories = ["models", "data", "utils"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Diretório '{directory}' criado")
        else:
            print(f"✅ Diretório '{directory}' já existe")

def main():
    """Função principal"""
    print("🚀 Setup do Projeto de Classificação de Pneumonia")
    print("=" * 50)
    
    # Verificar versão do Python
    if not check_python_version():
        sys.exit(1)
    
    # Criar diretórios
    create_directories()
    
    # Criar ambiente virtual
    if not create_virtual_environment():
        print("❌ Falha ao criar ambiente virtual!")
        sys.exit(1)
    
    # Instruções de ativação
    activate_virtual_environment()
    
    print("\n" + "=" * 50)
    print("🎉 Setup concluído!")
    print("\n📋 Próximos passos:")
    print("1. Ative o ambiente virtual:")
    if platform.system() == "Windows":
        print("   pneumonia_env\\Scripts\\activate")
    else:
        print("   source pneumonia_env/bin/activate")
    
    print("2. Instale as dependências:")
    print("   pip install -r requirements.txt")
    
    print("3. Treine o modelo:")
    print("   python train.py")
    
    print("4. Execute o app:")
    print("   streamlit run app.py")
    
    print("\n📚 Para mais informações, consulte o README.md")

if __name__ == "__main__":
    main() 