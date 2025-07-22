from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from fastapi import Body
from typing import Dict, List
import uuid

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    response: str

# Simple in-memory session store
session_memories: Dict[str, List[Dict[str, str]]] = {}

# RAG setup (placeholder for future document indexing feature)
# DOCS_PATH = "docs.txt"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 
# try:
#     loader = TextLoader(DOCS_PATH)
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = text_splitter.split_documents(documents)
#     embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     retriever = vectorstore.as_retriever()
#     print("[RAG] Document indexing enabled")
# except Exception as e:
#     print(f"[RAG] Document indexing disabled: {e}")
#     retriever = None

class RAGQueryRequest(BaseModel):
    query: str
    session_id: str = None
    max_tokens: int = 256
    temperature: float = 0.7

class RAGQueryResponse(BaseModel):
    response: str
    session_id: str
    conversation_history: List[Dict[str, str]]

def create_app():
    MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH", "./mistral-7b-instruct-v0.2.Q2_K.gguf")
    try:
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count(), n_gpu_layers=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

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

    @app.post("/rag_query", response_model=RAGQueryResponse)
    def rag_query(req: RAGQueryRequest = Body(...)):
        try:
            # Generate session ID if not provided
            if req.session_id is None:
                req.session_id = str(uuid.uuid4())
            
            # Get or create session memory
            if req.session_id not in session_memories:
                session_memories[req.session_id] = []
            
            conversation_history = session_memories[req.session_id]
            
            # Build conversation context
            if conversation_history:
                # Include recent conversation history (last 5 exchanges to avoid token limits)
                recent_history = conversation_history[-10:]  # Last 10 messages
                context = "\n".join([
                    f"{'User' if i % 2 == 0 else 'Assistant'}: {msg['content']}"
                    for i, msg in enumerate(recent_history)
                ])
                prompt = f"Previous conversation:\n{context}\n\nUser: {req.query}\nAssistant:"
            else:
                prompt = f"User: {req.query}\nAssistant:"
            
            # TODO: Future enhancement - Add document retrieval here
            # if retriever is not None:
            #     retrieved_docs = retriever.get_relevant_documents(req.query)
            #     doc_context = "\n".join([d.page_content for d in retrieved_docs])
            #     prompt = f"Context from documents:\n{doc_context}\n\nPrevious conversation:\n{context}\n\nUser: {req.query}\nAssistant:"
            
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
            conversation_history.append({"role": "user", "content": req.query})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Keep only last 20 messages to prevent memory bloat
            if len(conversation_history) > 20:
                session_memories[req.session_id] = conversation_history[-20:]
            
            return RAGQueryResponse(
                response=response,
                session_id=req.session_id,
                conversation_history=conversation_history
            )
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

    return app

app = create_app() 