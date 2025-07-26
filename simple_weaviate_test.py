#!/usr/bin/env python3
"""
Simple test script for Weaviate connection
"""

import requests
import json

def test_weaviate_http():
    """Test Weaviate connection via HTTP"""
    print("🔍 Testing Weaviate HTTP connection...")
    
    try:
        # Test basic connection
        response = requests.get("http://localhost:8080/v1/meta")
        if response.status_code == 200:
            print("✅ Weaviate HTTP connection successful")
            meta = response.json()
            print(f"📊 Weaviate version: {meta.get('version', 'Unknown')}")
            return True
        else:
            print(f"❌ Weaviate HTTP connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Weaviate HTTP connection failed: {e}")
        return False

def test_weaviate_schema():
    """Test Weaviate schema"""
    print("\n🔍 Testing Weaviate schema...")
    
    try:
        response = requests.get("http://localhost:8080/v1/schema")
        if response.status_code == 200:
            schema = response.json()
            classes = schema.get('classes', [])
            print(f"✅ Schema retrieved successfully")
            print(f"📊 Number of classes: {len(classes)}")
            
            for cls in classes:
                print(f"   - {cls.get('class', 'Unknown')}")
            
            return True
        else:
            print(f"❌ Schema retrieval failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Schema retrieval failed: {e}")
        return False

def main():
    """Run simple tests"""
    print("🚀 Starting simple Weaviate tests...\n")
    
    # Test 1: HTTP connection
    if test_weaviate_http():
        # Test 2: Schema
        test_weaviate_schema()
    
    print("\n🎉 Simple tests completed!")

if __name__ == "__main__":
    main() 