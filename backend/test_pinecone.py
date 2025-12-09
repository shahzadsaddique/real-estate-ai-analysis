#!/usr/bin/env python3
"""
Diagnostic script to test Pinecone initialization.

Run this to check if Pinecone is configured correctly:
    python test_pinecone.py
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from config import settings
from services.pinecone_service import pinecone_service

def test_pinecone():
    """Test Pinecone initialization and basic operations."""
    print("=" * 60)
    print("Pinecone Configuration Test")
    print("=" * 60)
    
    # Check configuration
    print(f"\n1. Checking configuration...")
    print(f"   PINECONE_API_KEY: {'*' * 20 if settings.pinecone_api_key else 'NOT SET'}")
    print(f"   PINECONE_ENVIRONMENT: {settings.pinecone_environment}")
    print(f"   PINECONE_INDEX_NAME: {settings.pinecone_index_name}")
    
    if not settings.pinecone_api_key:
        print("\n❌ ERROR: PINECONE_API_KEY is not set!")
        print("   Set it in your .env file or environment variables.")
        return False
    
    if not settings.pinecone_environment:
        print("\n❌ ERROR: PINECONE_ENVIRONMENT is not set!")
        print("   Set it in your .env file or environment variables.")
        return False
    
    # Test initialization
    print(f"\n2. Testing Pinecone initialization...")
    try:
        pinecone_service.initialize()
        print("   ✅ Pinecone service initialized successfully!")
    except Exception as e:
        print(f"   ❌ Failed to initialize Pinecone: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test index connection
    print(f"\n3. Testing index connection...")
    try:
        stats = pinecone_service.get_index_stats()
        print(f"   ✅ Index connected successfully!")
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Dimension: {stats.get('dimension', 0)}")
    except Exception as e:
        print(f"   ❌ Failed to connect to index: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All Pinecone tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_pinecone()
    sys.exit(0 if success else 1)
