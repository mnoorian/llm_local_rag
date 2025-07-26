#!/bin/bash

# Agentic RAG for Financial Risk Analysis - Startup Script

echo "🚀 Starting Agentic RAG for Financial Risk Analysis"
echo "=================================================="

# Check if model file exists
MODEL_FILE="mistral-7b-instruct-v0.2.Q2_K.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model file not found: $MODEL_FILE"
    echo ""
    echo "📥 Please download the model file:"
    echo "wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    echo ""
    echo "Or visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    echo "and download a quantized version (Q4_K_M recommended)"
    exit 1
fi

echo "✅ Model file found: $MODEL_FILE"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "✅ Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

echo "✅ docker-compose is available"

# Create documents directory if it doesn't exist
mkdir -p documents

echo ""
echo "🔧 Starting services with Docker Compose..."
echo "This may take a few minutes on first run..."

# Start the services
docker-compose up --build -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo ""
echo "🔍 Checking service status..."

# Check Weaviate
if curl -s http://localhost:8080/v1/meta > /dev/null; then
    echo "✅ Weaviate is running on http://localhost:8080"
else
    echo "❌ Weaviate is not responding"
fi

# Check FastAPI backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ FastAPI backend is running on http://localhost:8000"
else
    echo "❌ FastAPI backend is not responding"
fi

# Check Streamlit UI
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Streamlit UI is running on http://localhost:8501"
else
    echo "❌ Streamlit UI is not responding"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📱 Access your application:"
echo "   • Streamlit UI: http://localhost:8501"
echo "   • FastAPI Backend: http://localhost:8000"
echo "   • Weaviate: http://localhost:8080"
echo ""
echo "🔧 Useful commands:"
echo "   • View logs: docker-compose logs -f"
echo "   • Stop services: docker-compose down"
echo "   • Restart services: docker-compose restart"
echo ""
echo "🧪 Test the system:"
echo "   • Run test script: python test_weaviate.py"
echo "   • Try example queries in the UI"
echo ""
echo "📚 Example queries to try:"
echo "   • 'What are the capital requirements under Basel III?'"
echo "   • 'How does Dodd-Frank affect proprietary trading?'"
echo "   • 'What are the key risk indicators for operational risk?'" 