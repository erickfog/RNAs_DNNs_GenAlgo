#!/bin/bash
# Script de inicialização para o Render

echo "🚀 Iniciando aplicação no Render..."

# Configurar variáveis de ambiente
export PORT=${PORT:-8501}
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

echo "📡 Configurações:"
echo "   Porta: $PORT"
echo "   Endereço: $STREAMLIT_SERVER_ADDRESS"
echo "   Headless: $STREAMLIT_SERVER_HEADLESS"

# Verificar se as dependências estão instaladas
echo "📦 Verificando dependências..."
python3 -c "import streamlit, numpy, PIL" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Dependências OK"
else
    echo "❌ Dependências faltando. Instalando..."
    pip install -r requirements-deploy.txt
fi

# Iniciar o Streamlit
echo "🚀 Iniciando Streamlit..."
streamlit run app.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
