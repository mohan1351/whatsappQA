"""
WhatsApp Message Embeddings with Sentence Transformers
Creates vector embeddings and stores in ChromaDB
Uses all-mpnet-base-v2 (768 dimensions, free, self-hosted)
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from datetime import datetime
from pathlib import Path
import sys

# Import preprocessing function from utility
try:
    from utility.whatsapp_chat_preprocessing import whatsapp_chat_preprocessing
except ImportError:
    print("⚠️  Warning: whatsapp_chat_preprocessing not found, using original messages")
    def whatsapp_chat_preprocessing(text):
        return text


def create_embeddings(folder_name, file_name):
    """
    Create embeddings for WhatsApp messages
    
    Args:
        folder_name: Folder name in data/whatsapp/ (e.g., 'parivar', 'vellapanti')
        file_name: File name without extension (e.g., 'parivar', 'vellapanti')
    
    Example:
        create_embeddings('parivar', 'parivar')
        # Processes: data/whatsapp/parivar/parivar_features.csv
    """
    
    print("="*80)
    print(f"WHATSAPP EMBEDDINGS - Processing: {folder_name}/{file_name}")
    print("="*80)
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    
    print("\n[1/5] Loading WhatsApp data...")
    csv_path = f"data/whatsapp/{folder_name}/{file_name}_features.csv"
    
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {csv_path}")
        print(f"   Please run steps 1-4 first to create the features CSV")
        return False
    
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
    cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
    print(f"   Cache location: {cache_dir}")
    
    # Load model (auto-downloads on first run, then loads from cache)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    print("✅ Model loaded successfully")
    
    # ============================================================================
    # 3. CHECK IF EMBEDDINGS ALREADY EXIST
    # ============================================================================
    
    print("\n[3/5] Checking for existing embeddings...")
    
    # Create persistent ChromaDB client
    db_path = f"./chroma_db_{folder_name}"
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Check if collection already exists
    collection_name = f"whatsapp_{folder_name}"
    existing_collections = [c.name for c in chroma_client.list_collections()]
    
    if collection_name in existing_collections:
        print(f"⚠️  Found existing collection: {collection_name}")
        collection = chroma_client.get_collection(collection_name)
        print(f"   Stored embeddings: {collection.count()}")
        
        response = input(f"\n   Recreate embeddings? This will DELETE existing data (y/n): ")
        if response.lower() != 'y':
            print("   Skipping - using existing embeddings")
            print(f"\n✅ Embeddings ready at: {db_path}")
            return True
        
        # Delete existing collection
        chroma_client.delete_collection(collection_name)
        print("   Deleted existing collection")
    else:
        print(f"   No existing collection found")
    
    print(f"   Creating embeddings for {len(df_with_text)} messages...")
    print(f"   This may take 1-3 minutes...")
    
    # ============================================================================
    # 4. CREATE EMBEDDINGS
    # ============================================================================
    
    start_time = datetime.now()
    
    # Convert messages to list
    messages_original = df_with_text['message_english'].tolist()
    
    # Apply preprocessing to clean messages for embeddings
    print("\n[4/5] Preprocessing and creating embeddings...")
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
    print(f"   After preprocessing: {len(messages)} messages")
    
    # Create embeddings (batch processing for speed)
    embeddings = model.encode(
        messages,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Created {len(embeddings)} embeddings")
    print(f"   Shape: {embeddings.shape} (messages × dimensions)")
    print(f"   Time taken: {duration:.2f} seconds")
    print(f"   Speed: {len(embeddings)/duration:.0f} messages/second")
    
    # ============================================================================
    # 5. STORE IN CHROMADB
    # ============================================================================
    
    print("\n[5/5] Storing embeddings in ChromaDB...")
    
    # Create new collection with cosine similarity
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={
            "model": "all-mpnet-base-v2", 
            "dimensions": 768,
            "folder": folder_name,
            "file": file_name,
            "hnsw:space": "cosine"  # Use cosine similarity
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
    
    print(f"   Adding {len(embeddings)} embeddings in {total_batches} batch(es)...")
    
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
    
    print(f"\n✅ Stored {collection.count()} embeddings in ChromaDB")
    print(f"   Location: {db_path}")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    
    print("\n" + "="*80)
    print("✅ EMBEDDING CREATION COMPLETE")
    print("="*80)
    
    summary = f"""
Model Information:
├─ Name: all-mpnet-base-v2
├─ Dimensions: 768
├─ Messages embedded: {len(embeddings)}
├─ Time taken: {duration:.2f} seconds
├─ Speed: {len(embeddings)/duration:.0f} messages/second
└─ Storage: {db_path}

Collection:
├─ Name: {collection_name}
├─ Count: {collection.count()}
└─ Metadata: sender, date, year, month, birthday_person

Next Steps:
1. ✅ Embeddings created
2. ⏩ Run Q&A interface: python sp8_optimized_hybrid_search.py
"""
    
    print(summary)
    print("="*80)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) >= 3:
        folder = sys.argv[1]
        file = sys.argv[2]
    else:
        # Default: vellapanti
        folder = "vellapanti"
        file = "vellapanti"
        print(f"💡 Usage: python sp5_whatsapp_embeddings_sentence_transformers.py <folder_name> <file_name>")
        print(f"   Using default: {folder}/{file}")
    
    # Run embedding creation
    success = create_embeddings(folder, file)
    
    if success:
        print(f"\n🚀 Ready for Q&A!")
        print(f"   python sp8_optimized_hybrid_search.py")
