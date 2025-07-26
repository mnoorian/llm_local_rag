#!/usr/bin/env python3
"""
Isolated Weaviate Test Script
Tests Weaviate functionality independently of the main application
"""

import os
import sys
import requests
import json
import weaviate
from weaviate.classes.config import ConnectionParams
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import time

class WeaviateIsolatedTester:
    def __init__(self, weaviate_url="http://localhost:8080"):
        self.weaviate_url = weaviate_url
        self.client = None
        self.vectorstore = None
        self.test_class_name = "TestDocuments"
        
    def test_http_connection(self):
        """Test basic HTTP connection to Weaviate"""
        print("🔍 Testing HTTP connection...")
        try:
            response = requests.get(f"{self.weaviate_url}/v1/meta", timeout=10)
            if response.status_code == 200:
                meta = response.json()
                print(f"✅ HTTP connection successful")
                print(f"📊 Weaviate version: {meta.get('version', 'Unknown')}")
                print(f"📊 Hostname: {meta.get('hostname', 'Unknown')}")
                return True
            else:
                print(f"❌ HTTP connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ HTTP connection failed: {e}")
            return False
    
    def test_weaviate_client(self):
        """Test Weaviate Python client connection"""
        print("\n🔍 Testing Weaviate Python client...")
        try:
            connection_params = ConnectionParams.from_url(self.weaviate_url)
            self.client = weaviate.WeaviateClient(connection_params)
            
            # Test connection by getting schema
            schema = self.client.schema.get()
            print("✅ Weaviate Python client connected successfully")
            print(f"📊 Available classes: {len(schema.get('classes', []))}")
            
            # List existing classes
            for cls in schema.get('classes', []):
                print(f"   - {cls.get('class', 'Unknown')}")
            
            return True
        except Exception as e:
            print(f"❌ Weaviate client connection failed: {e}")
            return False
    
    def test_embeddings(self):
        """Test embedding model"""
        print("\n🔍 Testing embedding model...")
        try:
            embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            
            # Test embedding generation
            test_text = "This is a test document for financial risk analysis."
            embedding = embeddings.embed_query(test_text)
            
            print("✅ Embedding model loaded successfully")
            print(f"📊 Embedding dimension: {len(embedding)}")
            print(f"📊 Model: {embedding_model}")
            
            return embeddings
        except Exception as e:
            print(f"❌ Embedding model failed: {e}")
            return None
    
    def test_vectorstore_creation(self, embeddings):
        """Test vector store creation"""
        print("\n🔍 Testing vector store creation...")
        try:
            self.vectorstore = Weaviate(
                client=self.client,
                index_name=self.test_class_name,
                text_key="content",
                embedding=embeddings,
                by_text=False
            )
            
            print("✅ Vector store created successfully")
            print(f"📊 Index name: {self.test_class_name}")
            return True
        except Exception as e:
            print(f"❌ Vector store creation failed: {e}")
            return False
    
    def test_document_operations(self):
        """Test document operations (add, search, delete)"""
        print("\n🔍 Testing document operations...")
        
        # Create test documents
        test_documents = [
            "Basel III introduces higher capital requirements for banks to improve financial stability.",
            "The Dodd-Frank Act regulates financial institutions and protects consumers from abusive practices.",
            "Operational risk management is crucial for financial institutions to prevent losses.",
            "Market risk involves potential losses due to changes in market prices and rates.",
            "Credit risk assessment helps banks evaluate borrower default probabilities."
        ]
        
        try:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            
            # Create document objects
            from langchain.schema import Document
            docs = []
            for i, text in enumerate(test_documents):
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": f"test_doc_{i+1}.txt",
                        "type": "test",
                        "id": f"test_{i+1}"
                    }
                )
                docs.append(doc)
            
            # Split documents
            split_docs = text_splitter.split_documents(docs)
            print(f"📊 Created {len(split_docs)} document chunks")
            
            # Add documents to vector store
            self.vectorstore.add_documents(split_docs)
            print("✅ Documents added to vector store")
            
            # Wait a moment for indexing
            time.sleep(2)
            
            return True
        except Exception as e:
            print(f"❌ Document operations failed: {e}")
            return False
    
    def test_similarity_search(self):
        """Test similarity search functionality"""
        print("\n🔍 Testing similarity search...")
        
        test_queries = [
            "What are capital requirements?",
            "How does Dodd-Frank protect consumers?",
            "What is operational risk?",
            "Market risk management strategies",
            "Credit risk assessment methods"
        ]
        
        try:
            for query in test_queries:
                print(f"\n🔎 Query: '{query}'")
                docs = self.vectorstore.similarity_search(query, k=2)
                
                print(f"📊 Retrieved {len(docs)} documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"   {i}. Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"      Content: {doc.page_content[:80]}...")
            
            return True
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
            return False
    
    def test_metadata_filtering(self):
        """Test metadata filtering"""
        print("\n🔍 Testing metadata filtering...")
        try:
            # Test filtering by metadata
            from weaviate.classes.query import Filter
            
            # Create a filter for test documents
            filter_query = Filter.by_property("type").equal("test")
            
            # Search with filter
            docs = self.vectorstore.similarity_search(
                "risk management",
                k=3,
                filter=filter_query
            )
            
            print(f"✅ Metadata filtering successful")
            print(f"📊 Retrieved {len(docs)} documents with filter")
            
            return True
        except Exception as e:
            print(f"❌ Metadata filtering failed: {e}")
            return False
    
    def test_cleanup(self):
        """Clean up test data"""
        print("\n🔍 Cleaning up test data...")
        try:
            # Delete the test class
            if self.client:
                self.client.schema.delete_class(self.test_class_name)
                print("✅ Test class deleted successfully")
            return True
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
            return False
    
    def run_all_tests(self, cleanup=True):
        """Run all tests in sequence"""
        print("🚀 Starting isolated Weaviate tests...\n")
        print("=" * 50)
        
        results = []
        
        # Test 1: HTTP connection
        results.append(("HTTP Connection", self.test_http_connection()))
        
        # Test 2: Weaviate client
        if results[-1][1]:
            results.append(("Weaviate Client", self.test_weaviate_client()))
        
        # Test 3: Embeddings
        if results[-1][1]:
            embeddings = self.test_embeddings()
            results.append(("Embeddings", embeddings is not None))
        
        # Test 4: Vector store
        if results[-1][1]:
            results.append(("Vector Store", self.test_vectorstore_creation(embeddings)))
        
        # Test 5: Document operations
        if results[-1][1]:
            results.append(("Document Operations", self.test_document_operations()))
        
        # Test 6: Similarity search
        if results[-1][1]:
            results.append(("Similarity Search", self.test_similarity_search()))
        
        # Test 7: Metadata filtering
        if results[-1][1]:
            results.append(("Metadata Filtering", self.test_metadata_filtering()))
        
        # Cleanup
        if cleanup:
            results.append(("Cleanup", self.test_cleanup()))
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<20} {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Weaviate is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
        
        return passed == total

def main():
    """Main function"""
    # Check if Weaviate URL is provided as argument
    weaviate_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    print(f"🔗 Testing Weaviate at: {weaviate_url}")
    
    # Create tester and run tests
    tester = WeaviateIsolatedTester(weaviate_url)
    success = tester.run_all_tests(cleanup=True)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 