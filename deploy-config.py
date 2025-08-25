#!/usr/bin/env python3
"""
Configuração para deploy do projeto de classificação de pneumonia
"""

import os
import sys

def setup_deploy_environment():
    """Configura o ambiente para deploy"""
    
    # Configurações para deploy
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Desabilitar GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduzir logs do TensorFlow
    
    # Configurações do Streamlit para deploy
    os.environ['STREAMLIT_SERVER_PORT'] = os.environ.get('PORT', '8501')
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    print("🚀 Ambiente de deploy configurado!")
    print(f"   Porta: {os.environ['STREAMLIT_SERVER_PORT']}")
    print(f"   Endereço: {os.environ['STREAMLIT_SERVER_ADDRESS']}")

def check_deploy_requirements():
    """Verifica se as dependências de deploy estão disponíveis"""
    
    required_packages = [
        'streamlit',
        'numpy', 
        'pillow',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - NÃO DISPONÍVEL")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Pacotes faltando: {', '.join(missing_packages)}")
        print("   Execute: pip install -r requirements-deploy.txt")
        return False
    else:
        print("\n🎉 Todas as dependências de deploy estão disponíveis!")
        return True

def main():
    """Função principal"""
    print("🚀 Configuração de Deploy")
    print("=" * 40)
    
    # Configurar ambiente
    setup_deploy_environment()
    
    # Verificar dependências
    print("\n📦 Verificando dependências...")
    if check_deploy_requirements():
        print("\n✅ Ambiente pronto para deploy!")
        print("🚀 Execute: streamlit run app.py")
    else:
        print("\n❌ Ambiente não está pronto para deploy")
        sys.exit(1)

if __name__ == "__main__":
    main()
