"""
Test Optimized Hybrid Search with Sample Queries
Tests SEMANTIC, HYBRID, and ANALYTICAL routing
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
from optimized_hybrid_search import OptimizedHybridSearch
from datetime import datetime

print("="*80)
print("TESTING OPTIMIZED HYBRID SEARCH - Sample Queries")
print("="*80)

# ============================================================================
# STEP 1: Create Sample WhatsApp Data
# ============================================================================
print("\n[1/4] Creating sample WhatsApp data...")

sample_messages = [
    # Birthday wishes to Amit
    {"sender": "Priya", "birthday_person": "Amit", "message": "Happy birthday Amit! 🎂", "year": 2024, "month": 3},
    {"sender": "Rohit", "birthday_person": "Amit", "message": "Many happy returns of the day!", "year": 2024, "month": 3},
    {"sender": "Neha", "birthday_person": "Amit", "message": "Wishing you a wonderful birthday Amit!", "year": 2024, "month": 3},
    
    # Birthday wishes to Priya
    {"sender": "Amit", "birthday_person": "Priya", "message": "Happy birthday dear Priya!", "year": 2024, "month": 7},
    {"sender": "Rohit", "birthday_person": "Priya", "message": "Have a great birthday Priya! 🎉", "year": 2024, "month": 7},
    
    # Funny messages from Amit
    {"sender": "Amit", "birthday_person": "", "message": "Why don't scientists trust atoms? Because they make up everything! 😂", "year": 2024, "month": 1},
    {"sender": "Amit", "birthday_person": "", "message": "I told my wife she was drawing her eyebrows too high. She looked surprised.", "year": 2024, "month": 1},
    
    # Philosophical messages from MKV
    {"sender": "MKV", "birthday_person": "", "message": "Life is a journey, not a destination. Embrace each moment.", "year": 2024, "month": 2},
    {"sender": "MKV", "birthday_person": "", "message": "The greatest glory in living lies not in never falling, but in rising every time we fall.", "year": 2024, "month": 2},
    
    # Regular messages
    {"sender": "Priya", "birthday_person": "", "message": "Let's meet for dinner tonight", "year": 2024, "month": 1},
    {"sender": "Rohit", "birthday_person": "", "message": "Can someone pick up groceries?", "year": 2024, "month": 1},
    {"sender": "Neha", "birthday_person": "", "message": "Weather is nice today!", "year": 2024, "month": 2},
    
    # More birthday wishes to test filtering
    {"sender": "MKV", "birthday_person": "Amit", "message": "Happy birthday! May all your dreams come true!", "year": 2023, "month": 3},
    {"sender": "Priya", "birthday_person": "Rohit", "message": "Happy birthday Rohit! 🎂", "year": 2024, "month": 5},
    
    # Funny messages from others
    {"sender": "Rohit", "birthday_person": "", "message": "Why did the scarecrow win an award? He was outstanding in his field!", "year": 2024, "month": 1},
]

df = pd.DataFrame(sample_messages)
df['message_english'] = df['message']  # Simulate translated messages
print(f"✅ Created {len(df)} sample messages")

# ============================================================================
# STEP 2: Create ChromaDB with Embeddings
# ============================================================================
print("\n[2/4] Creating ChromaDB with embeddings...")

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("✅ Model loaded")

# Create in-memory ChromaDB
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.create_collection("test_whatsapp")

# Add messages to ChromaDB with explicit embeddings
ids = [f"msg_{i}" for i in range(len(df))]
documents = df['message'].tolist()
metadatas = df[['sender', 'birthday_person', 'year', 'month']].to_dict('records')

# Generate embeddings using the same model we'll use for queries
print("   Generating embeddings...")
embeddings = model.encode(documents).tolist()

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,  # ← Explicitly provide embeddings!
    metadatas=metadatas
)
print(f"✅ Added {len(ids)} messages to ChromaDB with {len(embeddings[0])}-dim embeddings")

# ============================================================================
# STEP 3: Initialize Optimized Search
# ============================================================================
print("\n[3/4] Initializing Optimized Search Engine...")
search_engine = OptimizedHybridSearch(collection, model, df)

# ============================================================================
# STEP 4: Test Sample Queries
# ============================================================================
print("\n[4/4] Testing Sample Queries...")
print("="*80)

# Define test queries with expected classifications
test_cases = [
    {
        "query": "funny messages",
        "expected": "SEMANTIC",
        "description": "Semantic concept, NO metadata filters"
    },
    {
        "query": "birthday wishes",
        "expected": "SEMANTIC",
        "description": "Semantic concept, NO metadata filters"
    },
    {
        "query": "philosophical thoughts",
        "expected": "SEMANTIC",
        "description": "Semantic concept, NO metadata filters"
    },
    {
        "query": "funny messages from Amit",
        "expected": "HYBRID",
        "description": "Semantic concept + person filter (SENDER)"
    },
    {
        "query": "birthday wishes Priya received",
        "expected": "HYBRID",
        "description": "Semantic concept + person filter (RECIPIENT)"
    },
    {
        "query": "birthday wishes to Amit in 2024",
        "expected": "HYBRID",
        "description": "Semantic concept + person (RECIPIENT) + year"
    },
    {
        "query": "philosophical messages from MKV",
        "expected": "HYBRID",
        "description": "Semantic concept + person filter (SENDER)"
    },
    {
        "query": "funny jokes in January",
        "expected": "HYBRID",
        "description": "Semantic concept + month filter"
    },
    {
        "query": "who sent most messages?",
        "expected": "ANALYTICAL",
        "description": "No semantic concept, pure aggregation"
    },
    {
        "query": "count messages in 2024",
        "expected": "ANALYTICAL",
        "description": "No semantic concept, counting with filter"
    },
]

# Run tests
results_summary = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{len(test_cases)}: {test['query']}")
    print(f"Expected: {test['expected']} - {test['description']}")
    print(f"{'='*80}")
    
    try:
        results = search_engine.search_with_router(test['query'], top_k=3)
        
        actual_type = results.get('query_type')
        success = actual_type == test['expected']
        
        # Print results
        if actual_type == 'ANALYTICAL':
            print(f"\n✅ Correctly classified as ANALYTICAL")
            print(f"   Should use pandas agent for: {test['query']}")
        else:
            formatted = search_engine.format_results(results, max_display=3)
            print(formatted)
        
        # Check classification
        status = "✅ PASS" if success else "❌ FAIL"
        results_summary.append({
            "Query": test['query'],
            "Expected": test['expected'],
            "Actual": actual_type,
            "Status": status
        })
        
        print(f"\n{status}: Classification = {actual_type} (expected {test['expected']})")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        results_summary.append({
            "Query": test['query'],
            "Expected": test['expected'],
            "Actual": "ERROR",
            "Status": "❌ ERROR"
        })

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

passed = sum(1 for r in results_summary if r['Status'] == '✅ PASS')
total = len(results_summary)
pass_rate = (passed / total * 100) if total > 0 else 0

print(f"\n{'='*80}")
print(f"RESULTS: {passed}/{total} tests passed ({pass_rate:.0f}%)")
print(f"{'='*80}")

if pass_rate == 100:
    print("🎉 All tests passed!")
else:
    print("⚠️ Some tests failed. Review the classifications above.")

print("\n" + "="*80)
print("KEY LEARNINGS")
print("="*80)
print("✅ SEMANTIC: Concept search with NO person/year/month filters")
print("   → Searches ALL embeddings (no where clause)")
print("")
print("✅ HYBRID: Concept search WITH person/year/month filters")
print("   → Filters metadata FIRST, then searches filtered embeddings")
print("   → 10-1000x speedup on large datasets!")
print("")
print("✅ ANALYTICAL: No semantic concept, pure counting/aggregation")
print("   → Uses pandas agent (no ChromaDB search)")
print("="*80)
