"""
Test Negation Query with Pandas Agent
Shows generated pandas code for analytical queries
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from optimized_hybrid_search import OptimizedHybridSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_negation_queries():
    """
    Test negation queries to see generated pandas code
    """
    print("="*80)
    print("NEGATION QUERY TEST - Pandas Code Generation")
    print("="*80)
    
    # Load WhatsApp data
    print("\n[1/4] Loading WhatsApp data...")
    df = pd.read_csv("data/whatsapp/parivar/parivar_features.csv", encoding="utf-8-sig")
    df_filtered = df[df['message_english'].notna() & (df['message_english'].str.len() >= 3)]
    print(f"✅ Loaded {len(df_filtered)} messages")
    
    # Load embedding model
    print("\n[2/4] Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("✅ Model loaded")
    
    # Connect to ChromaDB
    print("\n[3/4] Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db_sentence_transformers")
    collection = chroma_client.get_collection("whatsapp_messages_sentence_transformers")
    print(f"✅ Connected to ChromaDB ({collection.count()} embeddings)")
    
    # Initialize search engine with pandas agent
    print("\n[4/4] Initializing search engine with pandas agent...")
    search_engine = OptimizedHybridSearch(collection, model, df_filtered, use_pandas_agent=True)
    print("\n" + "="*80)
    
    # Test queries
    test_queries = [
        # Negation queries (ANALYTICAL)
        "who did not wish Krati birthday in 2025",
        "who didn't send messages in January",
        "list people who never sent birthday wishes",
        
        # Regular analytical queries
        "how many messages are there",
        "who sent most messages",
        "count messages by sender",
        
        # Hybrid query for comparison (no code)
        "funny messages from Amit in 2024"
    ]
    
    for query in test_queries:
        print(f"\n\n{'='*80}")
        print(f"🔍 Query: {query}")
        print("="*80)
        
        results = search_engine.search_with_router(query, top_k=5)
        print(search_engine.format_results(results))


if __name__ == "__main__":
    try:
        test_negation_queries()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
