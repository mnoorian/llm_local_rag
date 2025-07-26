from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from llama_cpp import Llama
import os
import json
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
from fastapi import Body
from typing import Dict, List, Optional, Any
import uuid
import re
from datetime import datetime
import asyncio
from bs4 import BeautifulSoup
import urllib.parse
import weaviate
from weaviate.embedded import EmbeddedOptions

# Enhanced data models for Agentic RAG
class AgenticQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    use_web_search: bool = True
    use_document_search: bool = True
    reasoning_steps: int = 3

class Citation(BaseModel):
    source: str
    content: str
    relevance_score: float
    source_type: str  # "document", "web", "regulatory"
    metadata: Dict[str, Any] = {}

class AgenticResponse(BaseModel):
    response: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    citations: List[Citation]
    reasoning_chain: List[str]
    confidence_score: float
    tools_used: List[str]

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    response: str

class RAGQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7

class RAGQueryResponse(BaseModel):
    response: str
    session_id: str
    conversation_history: List[Dict[str, str]]

# Global variables
session_memories: Dict[str, ConversationBufferMemory] = {}
weaviate_client = None
vectorstore = None
llm = None

# Financial regulatory documents and knowledge base
REGULATORY_DOCS = {
    "basel_iii": """
    Basel III is a comprehensive set of reform measures, developed by the Basel Committee on Banking Supervision, 
    to strengthen the regulation, supervision and risk management of the banking sector. The measures include:
    - Higher capital requirements
    - Introduction of leverage ratio
    - Liquidity requirements
    - Countercyclical capital buffer
    """,
    "dodd_frank": """
    The Dodd-Frank Wall Street Reform and Consumer Protection Act is a United States federal law that places 
    regulation of the financial industry in the hands of the government. Key provisions include:
    - Volcker Rule restrictions on proprietary trading
    - Enhanced supervision of systemically important financial institutions
    - Consumer protection measures
    - Derivatives regulation
    """,
    "solvency_ii": """
    Solvency II is a European Union directive that codifies and harmonizes EU insurance regulation. 
    It consists of three pillars:
    - Pillar 1: Quantitative requirements (capital requirements)
    - Pillar 2: Qualitative requirements (risk management)
    - Pillar 3: Reporting and disclosure requirements
    """
}

def initialize_weaviate():
    """Initialize Weaviate client and vector store"""
    global weaviate_client, vectorstore
    
    try:
        # Connect to Weaviate
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        weaviate_client = weaviate.Client(weaviate_url)
        
        # Initialize embeddings
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize vector store
        vectorstore = Weaviate(
            client=weaviate_client,
            index_name="FinancialDocuments",
            text_key="content",
            embedding=embeddings,
            by_text=False
        )
        
        print("✅ Weaviate initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Weaviate: {e}")
        return False

def load_documents_to_weaviate():
    """Load documents into Weaviate vector store"""
    global vectorstore
    
    if not vectorstore:
        print("❌ Vector store not initialized")
        return False
    
    try:
        # Load regulatory documents
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Load from financial_regulatory_docs.txt
        if os.path.exists("financial_regulatory_docs.txt"):
            loader = TextLoader("financial_regulatory_docs.txt")
            docs = loader.load()
            split_docs = text_splitter.split_documents(docs)
            for doc in split_docs:
                doc.metadata["source"] = "financial_regulatory_docs.txt"
                doc.metadata["type"] = "regulatory"
            documents.extend(split_docs)
        
        # Load from documents directory
        documents_dir = "documents"
        if os.path.exists(documents_dir):
            for file_path in os.listdir(documents_dir):
                if file_path.endswith('.txt'):
                    loader = TextLoader(os.path.join(documents_dir, file_path))
                    docs = loader.load()
                    split_docs = text_splitter.split_documents(docs)
                    for doc in split_docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["type"] = "document"
                    documents.extend(split_docs)
                elif file_path.endswith('.pdf'):
                    loader = PyPDFLoader(os.path.join(documents_dir, file_path))
                    docs = loader.load()
                    split_docs = text_splitter.split_documents(docs)
                    for doc in split_docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["type"] = "document"
                    documents.extend(split_docs)
        
        # Add documents to vector store
        if documents:
            vectorstore.add_documents(documents)
            print(f"✅ Loaded {len(documents)} document chunks to Weaviate")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load documents: {e}")
        return False

# Web search function (simulated for PoC)
def web_search(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Simulated web search - in production, use real search APIs"""
    mock_results = [
        {
            "title": f"Financial Risk Analysis: {query}",
            "snippet": f"Latest insights on {query} in financial markets and regulatory compliance.",
            "url": f"https://example.com/financial-risk/{query.replace(' ', '-')}",
            "source": "Financial Times"
        },
        {
            "title": f"Regulatory Update: {query}",
            "snippet": f"Recent regulatory changes affecting {query} and their impact on financial institutions.",
            "url": f"https://example.com/regulatory/{query.replace(' ', '-')}",
            "source": "Regulatory Digest"
        }
    ]
    return mock_results[:max_results]

# Document retrieval function using Weaviate
def retrieve_relevant_documents(query: str, k: int = 5) -> List[Citation]:
    """Retrieve relevant documents from Weaviate vector store"""
    global vectorstore
    
    citations = []
    
    if not vectorstore:
        # Fallback to in-memory search
        return retrieve_relevant_documents_fallback(query)
    
    try:
        # Search in vector store
        docs = vectorstore.similarity_search(query, k=k)
        
        for doc in docs:
            citations.append(Citation(
                source=doc.metadata.get("source", "Unknown"),
                content=doc.page_content,
                relevance_score=0.8,  # Weaviate similarity score
                source_type=doc.metadata.get("type", "document"),
                metadata=doc.metadata
            ))
        
        return citations
    except Exception as e:
        print(f"❌ Vector store search failed: {e}")
        return retrieve_relevant_documents_fallback(query)

def retrieve_relevant_documents_fallback(query: str) -> List[Citation]:
    """Fallback document retrieval using in-memory search"""
    citations = []
    query_lower = query.lower()
    
    for doc_name, content in REGULATORY_DOCS.items():
        relevance_score = 0
        keywords = query_lower.split()
        for keyword in keywords:
            if keyword in content.lower():
                relevance_score += 0.1
        
        if relevance_score > 0.1:
            citations.append(Citation(
                source=doc_name,
                content=content[:500] + "..." if len(content) > 500 else content,
                relevance_score=min(relevance_score, 1.0),
                source_type="regulatory"
            ))
    
    citations.sort(key=lambda x: x.relevance_score, reverse=True)
    return citations[:3]

# Reasoning function
def generate_reasoning_chain(query: str, context: str, llm) -> List[str]:
    """Generate step-by-step reasoning for the query"""
    reasoning_prompt = f"""
    You are a financial risk analyst assistant. Given the following query and context, 
    provide step-by-step reasoning to arrive at a comprehensive answer.
    
    Query: {query}
    Context: {context}
    
    Please provide 3-4 reasoning steps:
    1. 
    2. 
    3. 
    4. 
    """
    
    try:
        output = llm(
            reasoning_prompt,
            max_tokens=300,
            temperature=0.3,
            stop=None,
            echo=False
        )
        reasoning_text = output["choices"][0]["text"].strip()
        
        # Parse reasoning steps
        steps = []
        for line in reasoning_text.split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                steps.append(line.strip())
        
        return steps if steps else [reasoning_text]
    except Exception as e:
        return [f"Reasoning generation failed: {str(e)}"]

# Enhanced agentic query processing
def process_agentic_query(query: str, session_id: str, llm, use_web_search: bool = True, 
                         use_document_search: bool = True, reasoning_steps: int = 3) -> AgenticResponse:
    """Process query using agentic RAG approach"""
    
    # Step 1: Gather information from multiple sources
    citations = []
    tools_used = []
    
    if use_document_search:
        doc_citations = retrieve_relevant_documents(query)
        citations.extend(doc_citations)
        tools_used.append("document_search")
    
    if use_web_search:
        web_results = web_search(query)
        for result in web_results:
            citations.append(Citation(
                source=result["source"],
                content=result["snippet"],
                relevance_score=0.8,
                source_type="web"
            ))
        tools_used.append("web_search")
    
    # Step 2: Build comprehensive context
    context_parts = []
    if citations:
        context_parts.append("Relevant Information Sources:")
        for i, citation in enumerate(citations, 1):
            context_parts.append(f"{i}. {citation.source_type.upper()}: {citation.content}")
    
    context = "\n".join(context_parts) if context_parts else "No specific sources found."
    
    # Step 3: Generate reasoning chain
    reasoning_chain = generate_reasoning_chain(query, context, llm)
    
    # Step 4: Generate final response
    response_prompt = f"""
    You are an expert financial risk analyst assistant. Answer the following query based on the provided context and reasoning.
    
    Query: {query}
    
    Context and Sources:
    {context}
    
    Reasoning Steps:
    {chr(10).join(reasoning_chain)}
    
    Please provide a comprehensive, well-structured answer that:
    1. Directly addresses the query
    2. Incorporates relevant information from the sources
    3. Explains the reasoning behind your conclusions
    4. Includes specific citations where appropriate
    
    Answer:
    """
    
    try:
        output = llm(
            response_prompt,
            max_tokens=512,
            temperature=0.7,
            stop=None,
            echo=False
        )
        response = output["choices"][0]["text"].strip()
    except Exception as e:
        response = f"Error generating response: {str(e)}"
    
    # Step 5: Calculate confidence score
    confidence_score = min(0.9, 0.5 + len(citations) * 0.1 + len(reasoning_chain) * 0.05)
    
    return AgenticResponse(
        response=response,
        session_id=session_id,
        conversation_history=session_memories.get(session_id, ConversationBufferMemory()).chat_memory.messages,
        citations=citations,
        reasoning_chain=reasoning_chain,
        confidence_score=confidence_score,
        tools_used=tools_used
    )

def create_app():
    global llm
    
    MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "./mistral-7b-instruct-v0.2.Q2_K.gguf")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=os.cpu_count(),
            n_gpu_layers=0
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Initialize Weaviate
    if initialize_weaviate():
        load_documents_to_weaviate()
    else:
        print("⚠️ Weaviate not available, using fallback document search")

    app = FastAPI(title="Agentic RAG for Financial Risk Analysis")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": True,
            "weaviate_connected": weaviate_client is not None,
            "vectorstore_ready": vectorstore is not None
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        try:
            output = llm(
                req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=None,
                echo=False
            )
            return GenerateResponse(response=output["choices"][0]["text"].strip())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agentic_query", response_model=AgenticResponse)
    def agentic_query(req: AgenticQueryRequest = Body(...)):
        try:
            # Generate session ID if not provided
            if req.session_id is None:
                req.session_id = str(uuid.uuid4())
            
            # Get or create session memory
            if req.session_id not in session_memories:
                session_memories[req.session_id] = ConversationBufferMemory()
            
            # Process the query using agentic RAG
            result = process_agentic_query(
                query=req.query,
                session_id=req.session_id,
                llm=llm,
                use_web_search=req.use_web_search,
                use_document_search=req.use_document_search,
                reasoning_steps=req.reasoning_steps
            )
            
            # Update conversation history
            session_memories[req.session_id].chat_memory.add_user_message(req.query)
            session_memories[req.session_id].chat_memory.add_ai_message(result.response)
            
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/rag_query", response_model=RAGQueryResponse)
    def rag_query(req: RAGQueryRequest = Body(...)):
        try:
            # Generate session ID if not provided
            if req.session_id is None:
                req.session_id = str(uuid.uuid4())
            
            # Get or create session memory
            if req.session_id not in session_memories:
                session_memories[req.session_id] = ConversationBufferMemory()
            
            memory = session_memories[req.session_id]
            
            # Build conversation context
            conversation_history = memory.chat_memory.messages
            if conversation_history:
                recent_history = conversation_history[-10:]
                context = "\n".join([
                    f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}"
                    for i, msg in enumerate(recent_history)
                ])
                prompt = f"Previous conversation:\n{context}\n\nUser: {req.query}\nAssistant:"
            else:
                prompt = f"User: {req.query}\nAssistant:"
            
            # Generate response
            output = llm(
                prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=None,
                echo=False
            )
            response = output["choices"][0]["text"].strip()
            
            # Update conversation history
            memory.chat_memory.add_user_message(req.query)
            memory.chat_memory.add_ai_message(response)
            
            return RAGQueryResponse(
                response=response,
                session_id=req.session_id,
                conversation_history=[{"role": "user" if i % 2 == 0 else "assistant", "content": msg.content} 
                                   for i, msg in enumerate(memory.chat_memory.messages)]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/upload_document")
    async def upload_document(file: UploadFile = File(...)):
        """Upload and index a new document"""
        try:
            if not vectorstore:
                raise HTTPException(status_code=500, detail="Vector store not available")
            
            # Save uploaded file
            file_path = f"documents/{file.filename}"
            os.makedirs("documents", exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load and process document
            if file.filename.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file.filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Add metadata
            for doc in split_docs:
                doc.metadata["source"] = file.filename
                doc.metadata["type"] = "uploaded"
                doc.metadata["upload_time"] = datetime.now().isoformat()
            
            # Add to vector store
            vectorstore.add_documents(split_docs)
            
            return {
                "message": f"Document {file.filename} uploaded and indexed successfully",
                "chunks": len(split_docs)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/session/{session_id}")
    def clear_session(session_id: str):
        """Clear conversation history for a specific session"""
        if session_id in session_memories:
            del session_memories[session_id]
        return {"message": f"Session {session_id} cleared"}

    @app.get("/sessions")
    def list_sessions():
        """List all active sessions"""
        return {"sessions": list(session_memories.keys())}

    @app.get("/documents")
    def list_documents():
        """List indexed documents"""
        try:
            if vectorstore:
                # This would require a custom query to get document metadata
                return {"message": "Documents indexed in Weaviate"}
            else:
                return {"documents": list(REGULATORY_DOCS.keys())}
        except Exception as e:
            return {"error": str(e)}

    return app

app = create_app() 