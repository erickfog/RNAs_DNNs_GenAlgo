#!/usr/bin/env python3
"""
Script para limpar arquivos temporários e cache do projeto
"""

import os
import shutil
import glob

def cleanup_files():
    """Remove arquivos temporários e cache"""
    print("🧹 Limpeza do Projeto")
    print("=" * 30)
    
    # Arquivos para remover
    files_to_remove = [
        "*.pyc",
        "*.pyo",
        "__pycache__",
        "*.log",
        "*.tmp",
        "*.temp"
    ]
    
    # Diretórios para remover
    dirs_to_remove = [
        "__pycache__",
        ".pytest_cache",
        ".cache",
        "build",
        "dist",
        "*.egg-info"
    ]
    
    # Imagens temporárias (manter apenas as importantes)
    image_files = [
        "training_history.png",
        "confusion_matrix.png",
        "demo_results.png",
        "dataset_test_results.png"
    ]
    
    removed_count = 0
    
    # Remover arquivos
    for pattern in files_to_remove:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"🗑️  Removido: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Erro ao remover {file_path}: {e}")
    
    # Remover diretórios
    for pattern in dirs_to_remove:
        for dir_path in glob.glob(pattern):
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"🗑️  Removido diretório: {dir_path}")
                    removed_count += 1
            except Exception as e:
                print(f"⚠️  Erro ao remover diretório {dir_path}: {e}")
    
    # Remover imagens temporárias (exceto as importantes)
    for img_file in glob.glob("*.png"):
        if img_file not in image_files:
            try:
                os.remove(img_file)
                print(f"🗑️  Removida imagem: {img_file}")
                removed_count += 1
            except Exception as e:
                print(f"⚠️  Erro ao remover {img_file}: {e}")
    
    # Limpar cache do Python
    try:
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "-Bc", "import compileall; compileall.compile_dir('.', force=True)"], 
                              capture_output=True, text=True)
        print("🧹 Cache Python limpo")
    except Exception as e:
        print(f"⚠️  Erro ao limpar cache Python: {e}")
    
    print(f"\n✅ Limpeza concluída! {removed_count} itens removidos.")
    
    # Mostrar espaço em disco
    try:
        total, used, free = shutil.disk_usage('.')
        print(f"\n💾 Espaço em disco:")
        print(f"   Total: {total // (1024**3):.1f} GB")
        print(f"   Usado: {used // (1024**3):.1f} GB")
        print(f"   Livre: {free // (1024**3):.1f} GB")
    except Exception as e:
        print(f"⚠️  Não foi possível verificar espaço em disco: {e}")

def main():
    """Função principal"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        cleanup_files()
    else:
        print("🧹 Script de Limpeza do Projeto")
        print("=" * 40)
        print("Este script remove:")
        print("  - Arquivos Python compilados (*.pyc)")
        print("  - Diretórios __pycache__")
        print("  - Arquivos de log e temporários")
        print("  - Imagens temporárias")
        print("  - Cache do Python")
        print("\n⚠️  ATENÇÃO: Esta operação não pode ser desfeita!")
        
        response = input("\n❓ Continuar? (y/N): ").strip().lower()
        if response in ['y', 'yes', 's', 'sim']:
            cleanup_files()
        else:
            print("❌ Operação cancelada.")

if __name__ == "__main__":
    main() 