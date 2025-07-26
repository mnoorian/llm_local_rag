#!/usr/bin/env python3
"""
Interactive Weaviate Test Script
Allows manual testing of Weaviate operations
"""

import os
import sys
import weaviate
from weaviate.classes.config import ConnectionParams
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class InteractiveWeaviateTester:
    def __init__(self, weaviate_url="http://localhost:8080"):
        self.weaviate_url = weaviate_url
        self.client = None
        self.vectorstore = None
        self.test_class_name = "InteractiveTest"
        
    def connect(self):
        """Connect to Weaviate"""
        try:
            print(f"🔗 Connecting to Weaviate at {self.weaviate_url}...")
            connection_params = ConnectionParams.from_url(self.weaviate_url)
            self.client = weaviate.WeaviateClient(connection_params)
            
            # Test connection
            schema = self.client.schema.get()
            print("✅ Connected successfully!")
            print(f"📊 Available classes: {len(schema.get('classes', []))}")
            
            # Initialize embeddings
            embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            
            # Create vector store
            self.vectorstore = Weaviate(
                client=self.client,
                index_name=self.test_class_name,
                text_key="content",
                embedding=embeddings,
                by_text=False
            )
            
            print("✅ Vector store initialized!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def add_document(self):
        """Add a document to the vector store"""
        print("\n📝 Adding document...")
        
        # Get document content from user
        print("Enter document content (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        content = "\n".join(lines)
        if not content.strip():
            print("❌ No content provided")
            return False
        
        # Get metadata
        source = input("Enter source name (e.g., document.txt): ").strip() or "manual_input.txt"
        doc_type = input("Enter document type (e.g., regulatory, document): ").strip() or "manual"
        
        try:
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": source,
                    "type": doc_type,
                    "added_by": "interactive_test"
                }
            )
            
            # Split if needed
            if len(content) > 1000:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                docs = text_splitter.split_documents([doc])
            else:
                docs = [doc]
            
            # Add to vector store
            self.vectorstore.add_documents(docs)
            print(f"✅ Added {len(docs)} document chunk(s)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
            return False
    
    def search_documents(self):
        """Search documents in the vector store"""
        print("\n🔍 Searching documents...")
        
        query = input("Enter search query: ").strip()
        if not query:
            print("❌ No query provided")
            return False
        
        try:
            k = int(input("Enter number of results (default 3): ") or "3")
        except ValueError:
            k = 3
        
        try:
            # Perform search
            docs = self.vectorstore.similarity_search(query, k=k)
            
            print(f"\n📊 Found {len(docs)} results for: '{query}'")
            print("-" * 50)
            
            for i, doc in enumerate(docs, 1):
                print(f"\n📄 Result {i}:")
                print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"   Type: {doc.metadata.get('type', 'Unknown')}")
                print(f"   Content: {doc.page_content}")
                print("-" * 30)
            
            return True
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return False
    
    def list_documents(self):
        """List all documents in the vector store"""
        print("\n📋 Listing documents...")
        
        try:
            # Get all documents (this is a simplified approach)
            # In a real scenario, you might want to use Weaviate's GraphQL API
            docs = self.vectorstore.similarity_search("", k=100)  # Get up to 100 docs
            
            if not docs:
                print("📭 No documents found")
                return True
            
            print(f"📊 Found {len(docs)} documents:")
            print("-" * 50)
            
            sources = {}
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
            
            for source, count in sources.items():
                print(f"   📄 {source}: {count} chunk(s)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to list documents: {e}")
            return False
    
    def delete_documents(self):
        """Delete documents from the vector store"""
        print("\n🗑️ Deleting documents...")
        
        print("⚠️ This will delete ALL documents in the test class!")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm != "yes":
            print("❌ Operation cancelled")
            return False
        
        try:
            # Delete the entire class
            self.client.schema.delete_class(self.test_class_name)
            print("✅ All documents deleted")
            
            # Recreate the class
            self.vectorstore = Weaviate(
                client=self.client,
                index_name=self.test_class_name,
                text_key="content",
                embedding=HuggingFaceEmbeddings(
                    model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                ),
                by_text=False
            )
            print("✅ Test class recreated")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete documents: {e}")
            return False
    
    def show_menu(self):
        """Show interactive menu"""
        print("\n" + "=" * 50)
        print("🔧 INTERACTIVE WEAVIATE TESTER")
        print("=" * 50)
        print("1. Add document")
        print("2. Search documents")
        print("3. List documents")
        print("4. Delete all documents")
        print("5. Show Weaviate info")
        print("0. Exit")
        print("-" * 50)
    
    def show_weaviate_info(self):
        """Show Weaviate information"""
        print("\n📊 Weaviate Information...")
        
        try:
            # Get schema
            schema = self.client.schema.get()
            classes = schema.get('classes', [])
            
            print(f"📊 Total classes: {len(classes)}")
            print(f"🔗 Weaviate URL: {self.weaviate_url}")
            print(f"📝 Test class: {self.test_class_name}")
            
            if classes:
                print("\n📋 Available classes:")
                for cls in classes:
                    class_name = cls.get('class', 'Unknown')
                    properties = cls.get('properties', [])
                    print(f"   - {class_name} ({len(properties)} properties)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to get info: {e}")
            return False
    
    def run_interactive(self):
        """Run interactive testing session"""
        print("🚀 Starting Interactive Weaviate Tester...")
        
        if not self.connect():
            print("❌ Cannot proceed without connection")
            return False
        
        while True:
            self.show_menu()
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                self.add_document()
            elif choice == "2":
                self.search_documents()
            elif choice == "3":
                self.list_documents()
            elif choice == "4":
                self.delete_documents()
            elif choice == "5":
                self.show_weaviate_info()
            else:
                print("❌ Invalid choice. Please try again.")
            
            input("\nPress Enter to continue...")

def main():
    """Main function"""
    weaviate_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    tester = InteractiveWeaviateTester(weaviate_url)
    tester.run_interactive()

if __name__ == "__main__":
    main() 