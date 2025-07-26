#!/usr/bin/env python3
"""
Test script for Weaviate vector database setup and document retrieval
"""

import os
import sys
import weaviate
from weaviate.classes.config import ConnectionParams
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

def test_weaviate_connection():
    """Test basic Weaviate connection"""
    print("🔍 Testing Weaviate connection...")
    
    try:
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        connection_params = ConnectionParams.from_url(weaviate_url)
        client = weaviate.WeaviateClient(connection_params)
        
        # Test connection
        schema = client.schema.get()
        print("✅ Weaviate connection successful")
        print(f"📊 Available classes: {len(schema.get('classes', []))}")
        
        return client
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        return None

def test_embeddings():
    """Test embedding model"""
    print("\n🔍 Testing embedding model...")
    
    try:
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Test embedding
        test_text = "This is a test document for financial risk analysis."
        embedding = embeddings.embed_query(test_text)
        
        print("✅ Embedding model loaded successfully")
        print(f"📊 Embedding dimension: {len(embedding)}")
        
        return embeddings
    except Exception as e:
        print(f"❌ Embedding model failed: {e}")
        return None

def test_vectorstore(client, embeddings):
    """Test vector store initialization"""
    print("\n🔍 Testing vector store...")
    
    try:
        vectorstore = Weaviate(
            client=client,
            index_name="FinancialDocuments",
            text_key="content",
            embedding=embeddings,
            by_text=False
        )
        
        print("✅ Vector store initialized successfully")
        return vectorstore
    except Exception as e:
        print(f"❌ Vector store initialization failed: {e}")
        return None

def test_document_loading(vectorstore):
    """Test document loading and indexing"""
    print("\n🔍 Testing document loading...")
    
    try:
        # Load test document
        if os.path.exists("financial_regulatory_docs.txt"):
            loader = TextLoader("financial_regulatory_docs.txt")
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Add metadata
            for doc in split_docs:
                doc.metadata["source"] = "financial_regulatory_docs.txt"
                doc.metadata["type"] = "regulatory"
            
            # Add to vector store
            vectorstore.add_documents(split_docs)
            
            print(f"✅ Loaded {len(split_docs)} document chunks")
            return True
        else:
            print("⚠️ No financial_regulatory_docs.txt found")
            return False
            
    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        return False

def test_document_retrieval(vectorstore):
    """Test document retrieval"""
    print("\n🔍 Testing document retrieval...")
    
    try:
        # Test query
        query = "What are the capital requirements under Basel III?"
        docs = vectorstore.similarity_search(query, k=3)
        
        print(f"✅ Retrieved {len(docs)} documents for query: '{query}'")
        
        for i, doc in enumerate(docs, 1):
            print(f"📄 Document {i}:")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Type: {doc.metadata.get('type', 'Unknown')}")
            print(f"   Content: {doc.page_content[:100]}...")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Document retrieval failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Weaviate and RAG system tests...\n")
    
    # Test 1: Weaviate connection
    client = test_weaviate_connection()
    if not client:
        print("❌ Cannot proceed without Weaviate connection")
        sys.exit(1)
    
    # Test 2: Embeddings
    embeddings = test_embeddings()
    if not embeddings:
        print("❌ Cannot proceed without embedding model")
        sys.exit(1)
    
    # Test 3: Vector store
    vectorstore = test_vectorstore(client, embeddings)
    if not vectorstore:
        print("❌ Cannot proceed without vector store")
        sys.exit(1)
    
    # Test 4: Document loading
    if test_document_loading(vectorstore):
        # Test 5: Document retrieval
        test_document_retrieval(vectorstore)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 