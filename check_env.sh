#!/bin/bash
# Script para verificar y configurar el entorno de la aplicación

echo "=========================================="
echo "Verificador de Entorno - Health App"
echo "=========================================="
echo ""

# Check Python
echo "Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ $PYTHON_VERSION instalado"
else
    echo "❌ Python 3 no encontrado. Por favor instale Python 3.10 o superior."
    exit 1
fi

# Check pip
echo ""
echo "Verificando pip..."
if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
    echo "✅ pip instalado"
else
    echo "❌ pip no encontrado. Por favor instale pip."
    exit 1
fi

# Check Docker
echo ""
echo "Verificando Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "✅ $DOCKER_VERSION instalado"
    
    # Check if Docker is running
    if docker info &> /dev/null; then
        echo "✅ Docker daemon está corriendo"
    else
        echo "⚠️  Docker está instalado pero el daemon no está corriendo"
        echo "   Inicie Docker Desktop o el servicio de Docker"
    fi
else
    echo "⚠️  Docker no encontrado (opcional pero recomendado)"
fi

# Check Docker Compose
echo ""
echo "Verificando Docker Compose..."
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1; then
    echo "✅ Docker Compose instalado"
else
    echo "⚠️  Docker Compose no encontrado (opcional)"
fi

echo ""
echo "=========================================="
echo "Resumen:"
echo "=========================================="
echo ""
echo "Su sistema está listo para ejecutar la aplicación."
echo ""
echo "Opciones de ejecución:"
echo "  1. Python local: streamlit run app.py"
echo "  2. Docker: docker build -t health-app . && docker run -p 8501:8501 health-app"
echo "  3. Docker Compose: docker-compose up"
echo ""
echo "Para instalar dependencias de Python: pip install -r requirements.txt"
echo ""
