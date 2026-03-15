"""
WhatsApp Message Embeddings with Sentence Transformers + LLM Query Router
Uses all-mpnet-base-v2 (768 dimensions, 63.3% MTEB score)
Free, self-hosted, no API costs
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from datetime import datetime
from whatsapp_chat_preprocessing import whatsapp_chat_preprocessing
from crossencoder_reranker import CrossEncoderReranker
import json
import re

# Import query tools
from semantic_search_tool import semantic_search, semantic_search_simple
from analytical_queries import (
    count_messages_by_sender, 
    count_messages_by_time, 
    get_statistics,
    extract_year_from_query,
    extract_month_from_query
)
from set_operations_tool import (
    find_senders_who_mentioned,
    find_senders_who_did_not_mention,
    find_birthday_messages_for_person,
    extract_keyword_from_query,
    extract_person_name_from_query
)
from hybrid_search_tool import (
    hybrid_search,
    extract_sender_from_query
)
from llm_function_router import select_function_with_llm, AVAILABLE_FUNCTIONS

print("="*80)
print("WHATSAPP EMBEDDINGS - SENTENCE TRANSFORMERS + LLM ROUTER")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[1/5] Loading WhatsApp data...")
df = pd.read_csv("data/whatsapp/parivar/parivar_features.csv", encoding="utf-8-sig")

# Filter only messages with text
df_with_text = df[df['message_english'].notna() & (df['message_english'].str.strip() != '')]

# Filter out junk messages
junk_patterns = [
    '<Media omitted>',
    'This message was deleted',
    'image omitted',
    'video omitted',
    'audio omitted',
    'sticker omitted',
    'GIF omitted',
    'document omitted',
    'null'
]

print(f"   Before filtering: {len(df_with_text)} messages")

# Remove junk messages (case-insensitive)
for pattern in junk_patterns:
    df_with_text = df_with_text[~df_with_text['message_english'].str.lower().str.contains(pattern.lower(), na=False)]

# Remove messages that are only emojis or very short (< 3 characters)
df_with_text = df_with_text[df_with_text['message_english'].str.len() >= 3]

print(f"✅ Loaded {len(df_with_text)} actual messages (filtered out junk)")
print(f"   Columns: {list(df_with_text.columns)}")
print(f"   Date range: {df_with_text['dt'].min()} to {df_with_text['dt'].max()}")

# ============================================================================
# 2. LOAD SENTENCE TRANSFORMER MODEL
# ============================================================================

print("\n[2/5] Loading Sentence Transformer model...")
print("   Model: all-mpnet-base-v2")
print("   Dimensions: 768")
print("   MTEB Score: 63.3%")
print("   Size: ~420MB")
print("   First run: Downloads from Hugging Face")
print("   Subsequent runs: Loads from cache")

# Get cache directory
from pathlib import Path
cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
print(f"   Cache location: {cache_dir}")

# Load model (auto-downloads on first run, then loads from cache)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print("✅ Model loaded successfully from cache" if cache_dir.exists() else "✅ Model downloaded and loaded")

# ============================================================================
# 3. CHECK IF EMBEDDINGS ALREADY EXIST
# ============================================================================

print("\n[3/5] Checking for existing embeddings...")

# Create persistent ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db_sentence_transformers",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Check if collection already exists
collection_name = "whatsapp_messages_sentence_transformers"
existing_collections = [c.name for c in chroma_client.list_collections()]

if collection_name in existing_collections:
    print(f"✅ Found existing collection: {collection_name}")
    collection = chroma_client.get_collection(collection_name)
    print(f"   Stored embeddings: {collection.count()}")
    print(f"   Skipping embedding creation (already exists)")
    skip_embedding_creation = True
else:
    print(f"   No existing collection found")
    print(f"   Creating embeddings for all messages...")
    print(f"   This may take 1-2 minutes for 10K messages...")
    skip_embedding_creation = False

# ============================================================================
# 4. CREATE EMBEDDINGS (if needed)
# ============================================================================

if not skip_embedding_creation:
    start_time = datetime.now()

    # Convert messages to list
    messages_original = df_with_text['message_english'].tolist()

    # Apply preprocessing to clean messages for embeddings (original CSV unchanged!)
    print("   Preprocessing messages for better embeddings...")
    messages = []
    valid_indices = []

    for idx, msg in enumerate(messages_original):
        cleaned = whatsapp_chat_preprocessing(msg)
        if cleaned:  # Only keep messages that pass preprocessing
            messages.append(cleaned)
            valid_indices.append(idx)

    # Update dataframe to keep only valid messages
    df_with_text = df_with_text.iloc[valid_indices].reset_index(drop=True)

    print(f"   Original: {len(messages_original)} messages")
    print(f"   After preprocessing: {len(messages)} messages ({len(messages_original) - len(messages)} filtered)")

    # Create embeddings (batch processing for speed)
    embeddings = model.encode(
        messages,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"✅ Created {len(embeddings)} embeddings")
    print(f"   Shape: {embeddings.shape} (messages × dimensions)")
    print(f"   Time taken: {duration:.2f} seconds")
    print(f"   Speed: {len(embeddings)/duration:.0f} messages/second")

    # ============================================================================
    # 5. STORE IN CHROMADB
    # ============================================================================

    print("\n[4/5] Storing embeddings in ChromaDB...")

    # Delete collection if exists (for fresh start)
    try:
        chroma_client.delete_collection(collection_name)
        print("   Deleted existing collection for fresh start")
    except:
        pass

    # Create new collection with cosine similarity
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={
            "model": "all-mpnet-base-v2", 
            "dimensions": 768,
            "hnsw:space": "cosine"  # Use cosine similarity instead of L2 distance
        }
    )

    # Prepare data for ChromaDB
    ids = [f"msg_{i}" for i in range(len(df_with_text))]
    metadatas = []

    for idx, row in df_with_text.iterrows():
        metadata = {
            'sender': str(row['sender']),
            'date': str(row['dt']),
            'year': int(row['year']) if pd.notna(row['year']) else 0,
            'month': int(row['month']) if pd.notna(row['month']) else 0,
            'month_short': str(row['month_short']) if pd.notna(row['month_short']) else '',
            'birthday_person': str(row['birthday_person']) if pd.notna(row['birthday_person']) else '',
            'original_index': int(idx)
        }
        metadatas.append(metadata)

    # Add to collection in batches (ChromaDB max batch size = 5461)
    batch_size = 5000
    total_batches = (len(embeddings) + batch_size - 1) // batch_size

    print(f"   Adding {len(embeddings)} embeddings in {total_batches} batches...")

    for i in range(0, len(embeddings), batch_size):
        end_idx = min(i + batch_size, len(embeddings))
        batch_num = i // batch_size + 1
        
        collection.add(
            embeddings=embeddings[i:end_idx].tolist(),
            documents=messages[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        
        print(f"   Batch {batch_num}/{total_batches}: Added {end_idx - i} embeddings")

    print(f"✅ Stored {collection.count()} embeddings in ChromaDB")
    print(f"   Location: ./chroma_db_sentence_transformers")

# ============================================================================
# 5. TEST SEMANTIC SEARCH
# ============================================================================

print("\n[5/5] Testing semantic search...")
print("="*80)

# Import search functions from semantic_search_tool
from semantic_search_tool import search_messages, search_with_reranking

# Initialize CrossEncoder reranker
print("\n[Initializing CrossEncoder for reranking...]")
reranker = CrossEncoderReranker()


# Test Query 1: Without Reranking
print("\n" + "="*80)
print("🔍 Query 1: 'who appears a happy person' (WITHOUT Reranking)")
print("-"*80)

results = search_messages(collection, model, "who appears a happy person", n_results=5)

for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
), 1):
    # With cosine similarity, ChromaDB returns 1 - cosine_sim as distance
    # So similarity = 1 - distance gives us the actual cosine similarity
    similarity = 1 - distance  # Now properly bounded between -1 and 1
    print(f"\n{i}. [Similarity: {similarity:.3f}]")
    print(f"   Message: {doc[:100]}{'...' if len(doc) > 100 else ''}")
    print(f"   Sender: {metadata['sender']} | Date: {metadata['date']}")

# Test Query 2: Birthday wishes
print("\n" + "="*80)
print("🔍 Query 2: 'who talks about movie'")
print("-"*80)

results = search_messages(collection, model, "who talks about movie", n_results=5)

for i, (doc, metadata, distance) in enumerate(zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
), 1):
    similarity = 1 - distance
    print(f"\n{i}. [Similarity: {similarity:.3f}]")
    print(f"   Message: {doc[:100]}{'...' if len(doc) > 100 else ''}")
    print(f"   Sender: {metadata['sender']} | Date: {metadata['date']}")


# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================

print("\n" + "="*80)
print("✅ SENTENCE TRANSFORMERS SETUP COMPLETE")
print("="*80)

# Build summary (conditionally include metrics based on whether we created embeddings)
summary = f"""
Model Information:
├─ Name: all-mpnet-base-v2
├─ Dimensions: 768
├─ MTEB Score: 63.3%
├─ Cost: Free (self-hosted)"""

if not skip_embedding_creation:
    summary += f"""
└─ Storage: {embeddings.nbytes / (1024*1024):.2f} MB for embeddings

Performance:
├─ Messages embedded: {len(embeddings)}
├─ Time taken: {duration:.2f} seconds
├─ Speed: {len(embeddings)/duration:.0f} messages/second
└─ Storage location: ./chroma_db_sentence_transformers"""
else:
    summary += f"""
└─ Storage: ./chroma_db_sentence_transformers

Performance:
├─ Embeddings loaded: {collection.count()}
└─ Status: Used existing embeddings (no creation needed)"""

summary += """

Usage Example:
```python
# Load existing embeddings
chroma_client = chromadb.PersistentClient(path="./chroma_db_sentence_transformers")
collection = chroma_client.get_collection("whatsapp_messages")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Search
query_embedding = model.encode("your query here")
results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=10)
```

Next Steps:
1. ✅ Run this script to create embeddings
2. ⏳ Run OpenAI embeddings script (to compare)
3. ⏳ Evaluate precision of both models
4. ⏳ Choose best model for your needs
"""

print(summary)

# ============================================================================
# CROSSENCODER RERANKING COMPARISON
# ============================================================================

print("\n" + "="*80)
print("CROSSENCODER RERANKING COMPARISON")
print("="*80)
print("\nComparing: Embedding-only vs Two-Stage (Embedding + CrossEncoder)\n")

demo_query = "who appears a happy sad person"

# Stage 1: Embedding search only (baseline)
print(f"🔍 Query: '{demo_query}'")
print("\n--- Method 1: Embedding Search Only (Baseline) ---")
baseline_results = search_messages(collection, model, demo_query, n_results=50)

print(f"Top 5 results:")
baseline_data = []
for i, (doc, metadata, distance) in enumerate(zip(
    baseline_results['documents'][0][:10],
    baseline_results['metadatas'][0][:10],
    baseline_results['distances'][0][:10]
), 1):
    similarity = 1 - distance
    baseline_data.append({
        'method': 'Embedding Only',
        'rank': i,
        'score': similarity,
        'message': doc,
        'sender': metadata['sender'],
        'date': metadata['date']
    })
    if i <= 5:
        print(f"\n{i}. [Score: {similarity:.3f}]")
        print(f"   {doc[:80]}{'...' if len(doc) > 80 else ''}")
        print(f"   Sender: {metadata['sender']}")

# Stage 2: Two-stage search with reranking
print("\n" + "-"*80)
print("--- Method 2: Two-Stage Search (Embedding + CrossEncoder) ---")
print("Note: CrossEncoder scores converted to similarity (0-1 range)")
print("      Higher = more relevant, 0.5+ = good match, <0.01 = poor match")
reranked_results = search_with_reranking(collection, model, reranker, demo_query, n_candidates=50, top_k=10, use_similarity=True)

print(f"Top 5 results after reranking:")
reranked_data = []
for i, (doc, metadata, score) in enumerate(reranked_results[:10], 1):
    reranked_data.append({
        'method': 'Embedding + CrossEncoder',
        'rank': i,
        'score': score,
        'message': doc,
        'sender': metadata['sender'],
        'date': metadata['date']
    })
    if i <= 5:
        print(f"\n{i}. [Similarity: {score:.4f}]")
        print(f"   {doc[:80]}{'...' if len(doc) > 80 else ''}")
        print(f"   Sender: {metadata['sender']}")

print(f"\n💡 Interpretation:")
print(f"   - Similarity range: {reranked_results[0][2]:.4f} (best) to {reranked_results[-1][2]:.4f} (worst)")
print(f"   - Scores > 0.5 = strong matches")
print(f"   - Scores 0.1-0.5 = moderate matches")
print(f"   - Scores < 0.01 = poor matches (use min_score=0.01 to filter)")

# Save comparison to CSV
comparison_df = pd.DataFrame(baseline_data + reranked_data)
comparison_csv_path = "crossencoder_comparison_results.csv"
comparison_df.to_csv(comparison_csv_path, index=False, encoding='utf-8-sig')

print("\n" + "="*80)
print(f"✅ Comparison saved to: {comparison_csv_path}")
print("="*80)

# Create side-by-side comparison CSV
print("\n--- Creating side-by-side comparison CSV ---")
side_by_side = []
for rank in range(1, 11):
    baseline_row = baseline_data[rank - 1]
    reranked_row = reranked_data[rank - 1]
    
    side_by_side.append({
        'rank': rank,
        'query': demo_query,
        'embedding_only_score': baseline_row['score'],
        'embedding_only_message': baseline_row['message'][:100],
        'embedding_only_sender': baseline_row['sender'],
        'crossencoder_score': reranked_row['score'],
        'crossencoder_message': reranked_row['message'][:100],
        'crossencoder_sender': reranked_row['sender']
    })

side_by_side_df = pd.DataFrame(side_by_side)
side_by_side_csv_path = "crossencoder_side_by_side_comparison.csv"
side_by_side_df.to_csv(side_by_side_csv_path, index=False, encoding='utf-8-sig')

print(f"✅ Side-by-side comparison saved to: {side_by_side_csv_path}")

print("\n" + "="*80)
print("Expected Improvement: +15-25% precision with CrossEncoder reranking")
print("="*80)

# ============================================================================
# LLM QUERY ROUTER - Route questions to appropriate tools
# ============================================================================

print("\n" + "="*80)
print("LLM QUERY ROUTER - Multi-Tool Question Answering")
print("="*80)
# ============================================================================
# LLM QUERY ROUTER - Route questions to appropriate tools
# ============================================================================

print("\n" + "="*80)
print("LLM QUERY ROUTER - Multi-Tool Question Answering")
print("="*80)


def route_and_execute_query(question: str) -> dict:
    """
    Main router: Uses LLM to select function and executes it
    
    Args:
        question: Natural language question
    
    Returns:
        Dict with query type, results, and formatted answer
    """
    # Step 1: Use LLM Function Router to select function
    function_selection = select_function_with_llm(question)
    function_name = function_selection['function_name']
    params = function_selection['parameters']
    confidence = function_selection['confidence']
    
    print(f"\n📋 LLM Function Selection:")
    print(f"   Function: {function_name}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Parameters: {json.dumps(params, indent=6)}")
    
    # Step 2: Execute the selected function
    try:
        # Analytical Functions
        if function_name == 'get_statistics':
            result = get_statistics(df_with_text)
        
        elif function_name == 'count_messages_by_sender':
            result = count_messages_by_sender(
                df=df_with_text,
                year=params.get('year'),
                month=params.get('month'),
                top_n=params.get('top_n', 10)
            )
        
        elif function_name == 'count_messages_by_time':
            result = count_messages_by_time(
                df=df_with_text,
                group_by=params.get('group_by', 'month'),
                year=params.get('year')
            )
        
        # Set Operation Functions
        elif function_name == 'find_senders_who_mentioned':
            result = find_senders_who_mentioned(
                df=df_with_text,
                keyword=params.get('keyword', 'birthday'),
                year=params.get('year'),
                month=params.get('month')
            )
        
        elif function_name == 'find_senders_who_did_not_mention':
            result = find_senders_who_did_not_mention(
                df=df_with_text,
                keyword=params.get('keyword', 'birthday'),
                year=params.get('year'),
                month=params.get('month')
            )
        
        elif function_name == 'find_birthday_messages_for_person':
            result = find_birthday_messages_for_person(
                df=df_with_text,
                person_name=params.get('person_name', 'MKV'),
                year=params.get('year')
            )
        
        # Hybrid Search
        elif function_name == 'hybrid_search':
            result = hybrid_search(
                collection=collection,
                model=model,
                reranker=reranker,
                query=params.get('query', question),
                sender=params.get('sender'),
                year=params.get('year'),
                month=params.get('month'),
                top_k=params.get('top_k', 10)
            )
        
        # Semantic Search (default)
        else:  # function_name == 'semantic_search'
            result = semantic_search(
                collection=collection,
                model=model,
                reranker=reranker,
                query=params.get('query', question),
                top_k=params.get('top_k', 10)
            )
        
        return result
    
    except Exception as e:
        return {
            'query_type': 'error',
            'function_name': function_name,
            'error': str(e),
            'answer': f"❌ Error executing {function_name}: {str(e)}"
        }


# ============================================================================
# DEMO: Test different question types
# ============================================================================

print("\n" + "-"*80)
print("Testing LLM Router with different question types:")
print("-"*80)

test_questions = [
    "who sent max messages in 2024?",
    "who is more philosophical in life?",
    "who did not wish birthday to MKV?",
    "show me statistics",
    "philosophical messages from Atul in 2024"
]

for question in test_questions:
    print("\n" + "="*80)
    print(f"❓ Question: {question}")
    print("="*80)
    
    response = route_and_execute_query(question)
    
    print(f"\n{response['answer']}")

print("\n" + "="*80)
print("✅ LLM ROUTER READY!")
print("="*80)
print("\n💡 To ask custom questions:")
print("   response = route_and_execute_query('your question here')")
print("   print(response['answer'])")
print("="*80)
