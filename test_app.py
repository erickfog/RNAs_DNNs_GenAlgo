#!/usr/bin/env python3
"""
Script de teste simples para verificar se app.py está funcionando
"""

import ast
import sys

def test_syntax():
    """Testa se o arquivo app.py tem sintaxe válida"""
    try:
        with open('app.py', 'r', encoding='utf-8') as file:
            source = file.read()
        
        # Tentar compilar o código
        ast.parse(source)
        print("✅ Sintaxe do app.py está válida!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Erro de sintaxe no app.py: {e}")
        print(f"   Linha {e.lineno}, coluna {e.offset}")
        return False
        
    except Exception as e:
        print(f"❌ Erro ao ler/analisar app.py: {e}")
        return False

def test_imports():
    """Testa se os imports do app.py são válidos"""
    try:
        # Tentar importar o módulo
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("app", "app.py")
        if spec is None:
            print("❌ Não foi possível carregar app.py")
            return False
            
        print("✅ Imports do app.py estão válidos!")
        return True
        
    except Exception as e:
        print(f"❌ Erro nos imports do app.py: {e}")
        return False

def main():
    """Função principal"""
    print("🧪 Teste de Sintaxe do app.py")
    print("=" * 40)
    
    # Testar sintaxe
    syntax_ok = test_syntax()
    
    # Testar imports (apenas se a sintaxe estiver ok)
    imports_ok = False
    if syntax_ok:
        imports_ok = test_imports()
    
    # Resumo
    print("\n" + "=" * 40)
    print("📊 RESUMO DOS TESTES")
    print("=" * 40)
    
    print(f"Sintaxe: {'✅ OK' if syntax_ok else '❌ FALHOU'}")
    print(f"Imports: {'✅ OK' if imports_ok else '❌ FALHOU'}")
    
    if syntax_ok and imports_ok:
        print("\n🎉 app.py está funcionando corretamente!")
        print("🚀 Você pode executar: streamlit run app.py")
    else:
        print("\n⚠️  app.py tem problemas que precisam ser corrigidos.")
    
    return syntax_ok and imports_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 