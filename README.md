# Agentic RAG for Financial Risk Analysis

## 🚀 Overview

This is a Proof of Concept (PoC) for an **Agentic Retrieval-Augmented Generation (RAG) system** designed specifically for financial risk analysts. The system combines local LLM capabilities with Weaviate vector database, document retrieval, web search simulation, and multi-step reasoning to provide comprehensive answers to complex financial risk queries.

## 🎯 Key Features

### 🤖 Agentic Capabilities
- **Multi-step Reasoning**: Generates step-by-step reasoning chains for complex queries
- **Multi-tool Integration**: Combines document search, web search, and LLM reasoning
- **Citation Tracking**: Provides source citations with relevance scores
- **Confidence Scoring**: Estimates confidence levels for responses
- **Session Memory**: Maintains conversation context using LangChain's ConversationBufferMemory

### 📚 Knowledge Sources
- **Weaviate Vector Database**: Scalable vector storage for document embeddings
- **Regulatory Documents**: Basel III, Dodd-Frank, Solvency II frameworks
- **Document Upload**: Support for PDF and TXT files
- **Web Search**: Simulated web search for current information
- **Conversational Memory**: Context-aware responses based on conversation history

### 🛠️ Technical Features
- **Local LLM**: Runs Mistral 7B Instruct v0.2 locally using llama.cpp
- **Vector Database**: Weaviate with sentence-transformers embeddings
- **FastAPI Backend**: RESTful API with multiple endpoints
- **Streamlit UI**: Modern, interactive web interface with document upload
- **Docker Support**: Containerized deployment with Weaviate
- **Session Management**: Persistent conversation sessions using LangChain

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   Local LLM     │
│                 │◄──►│   Backend       │◄──►│   (Mistral 7B)  │
│ - Query Input   │    │                 │    │                 │
│ - Results Display│   │ - Agentic RAG   │    │ - Text Generation│
│ - Document Upload│   │ - Document Search│   │ - Reasoning     │
└─────────────────┘    │ - Web Search    │    └─────────────────┘
                       │ - Session Mgmt  │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Weaviate      │
                       │   Vector DB     │
                       │                 │
                       │ - Document      │
                       │   Embeddings    │
                       │ - Similarity    │
                       │   Search        │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- At least 8GB RAM (16GB recommended)
- Mistral 7B model file

### 1. Download the Model
```bash
# Download Mistral 7B Instruct v0.2 GGUF model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### 2. Setup with Docker
```bash
# Build and start services
docker-compose up --build

# Access the application
# UI: http://localhost:8501
# API: http://localhost:8000
# Weaviate: http://localhost:8080
```

### 3. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Weaviate (in separate terminal)
docker run -d -p 8080:8080 --name weaviate semitechnologies/weaviate:1.22.4

# Start backend
uvicorn main:app --host 0.0.0.0 --port 8000

# Start UI (new terminal)
streamlit run ui_streamlit.py
```

## 📖 Usage

### 1. Agentic RAG Mode (Recommended)
- **Multi-tool Analysis**: Combines document search, web search, and reasoning
- **Step-by-step Reasoning**: Shows the logical process behind answers
- **Citations**: Provides source references with relevance scores
- **Confidence Scoring**: Indicates reliability of responses
- **Document Upload**: Add custom financial documents

### 2. Conversational RAG Mode
- **Memory**: Maintains conversation context using LangChain
- **Document Search**: Searches regulatory documents in Weaviate
- **Context-Aware**: Builds on previous exchanges

### 3. Pure LLM Mode
- **Stateless**: Single-turn responses
- **Fast**: Direct LLM interaction
- **Simple**: Basic question-answering

## 🔍 Example Queries

### Regulatory Compliance
```
"What are the capital requirements under Basel III for credit risk?"
"How does Dodd-Frank affect proprietary trading?"
"What are the liquidity requirements under Solvency II?"
```

### Risk Assessment
```
"How should I assess counterparty credit risk in derivatives trading?"
"What are the key risk indicators for operational risk?"
"How do I calculate Value at Risk (VaR) for a portfolio?"
```

### Market Analysis
```
"What are the current trends in interest rate risk management?"
"How do I evaluate market risk in emerging markets?"
"What are the best practices for stress testing?"
```

## 🛠️ API Endpoints

### Agentic RAG
```http
POST /agentic_query
{
  "query": "What are Basel III capital requirements?",
  "use_web_search": true,
  "use_document_search": true,
  "reasoning_steps": 3
}
```

### Conversational RAG
```http
POST /rag_query
{
  "query": "Explain credit risk assessment",
  "session_id": "uuid"
}
```

### Basic LLM
```http
POST /generate
{
  "prompt": "What is financial risk?",
  "max_tokens": 256
}
```

### Document Upload
```http
POST /upload_document
Content-Type: multipart/form-data
file: [PDF or TXT file]
```

## ⚙️ Configuration

### Environment Variables
```bash
MISTRAL_MODEL_PATH=./mistral-7b-instruct-v0.2.Q4_K_M.gguf
WEAVIATE_URL=http://weaviate:8080
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Model Parameters
- **Context Length**: 4096 tokens
- **Threads**: Auto-detected CPU cores
- **GPU Layers**: 0 (CPU-only)
- **Temperature**: 0.1-1.0 (configurable)
- **Max Tokens**: 256-1024 (configurable)

## 📊 Performance Considerations

### Memory Usage
- **Model Loading**: ~4GB RAM for Q4_K_M quantization
- **Session Storage**: In-memory (consider Redis for production)
- **Document Indexing**: Weaviate vector store

### Response Time
- **Agentic Mode**: 10-30 seconds (multi-step processing)
- **RAG Mode**: 5-15 seconds (document search + generation)
- **LLM Mode**: 2-8 seconds (direct generation)

## 🔮 Future Enhancements

### Planned Features
- **Real Web Search**: Integration with search APIs
- **Advanced Reasoning**: Chain-of-thought and tree-of-thought
- **Multi-modal**: Support for charts and tables
- **Export Capabilities**: PDF reports and citations

### Production Considerations
- **Database**: PostgreSQL for session storage
- **Caching**: Redis for document embeddings
- **Authentication**: User management and access control
- **Monitoring**: Logging and performance metrics
- **Scaling**: Load balancing and horizontal scaling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mistral AI**: For the base model
- **llama.cpp**: For efficient local inference
- **LangChain**: For RAG framework
- **Weaviate**: For vector database
- **FastAPI**: For the backend framework
- **Streamlit**: For the web interface

---

**Note**: This is a Proof of Concept. For production use, implement proper security, authentication, and monitoring measures. 
