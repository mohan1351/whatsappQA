"""
Streamlit Q&A Application for WhatsApp Chat Analysis
Conversational interface for querying WhatsApp data using SP8 Optimized Hybrid Search
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from typing import Dict
import json

# Import SP8 components
from sp8_optimized_hybrid_search import OptimizedHybridSearch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="WhatsApp QA Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .chat-bot {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.hybrid_search = None
        st.session_state.df = None
        st.session_state.collection = None
        st.session_state.model = None
        st.session_state.folder_name = None
        st.session_state.error_message = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []


# ============================================================================
# LOAD DATA AND INITIALIZE SEARCH
# ============================================================================

@st.cache_resource
def load_model():
    """Load Sentence Transformer model (cached)"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


@st.cache_resource
def load_collection(folder_name: str):
    """Load ChromaDB collection (cached)"""
    db_path = f"./chroma_db_{folder_name}"
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    collection_name = f"whatsapp_{folder_name}"
    return chroma_client.get_collection(collection_name)


def load_hybrid_search(folder_name: str, file_name: str):
    """
    Load ChromaDB embeddings and initialize OptimizedHybridSearch
    Optimized with separate caching for model and collection
    
    Args:
        folder_name: Folder name (e.g., 'parivar', 'vellapanti')
        file_name: File name (e.g., 'parivar')
    
    Returns:
        OptimizedHybridSearch instance
    """
    try:
        # Step 1: Load DataFrame (fast)
        csv_path = f"data/whatsapp/{folder_name}/{file_name}_features.csv"
        
        if not Path(csv_path).exists():
            st.error(f"❌ File not found: {csv_path}")
            st.info("Please run main_pipeline.py first to create the features CSV")
            return None
        
        progress_bar = st.progress(0, text="📚 Loading messages...")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        st.success(f"✅ Loaded {len(df)} messages")
        progress_bar.progress(25)
        
        # Step 2: Load model (cached, ~30 seconds first time)
        progress_bar.progress(50, text="🤖 Loading Sentence Transformer model... (cached on next load)")
        model = load_model()
        progress_bar.progress(75)
        
        # Step 3: Load ChromaDB collection (cached)
        progress_bar.progress(85, text="🔍 Loading embeddings...")
        try:
            collection = load_collection(folder_name)
            st.success(f"✅ Loaded {collection.count()} embeddings")
        except Exception as e:
            st.error(f"❌ ChromaDB collection not found: {e}")
            st.info(f"Please run: python sp5_whatsapp_embeddings_sentence_transformers.py {folder_name} {file_name}")
            return None
        
        progress_bar.progress(95, text="⚙️ Initializing search...")
        
        # Step 4: Initialize OptimizedHybridSearch
        hybrid_search = OptimizedHybridSearch(
            collection=collection,
            model=model,
            df=df,
            folder_name=folder_name,
            use_pandas_agent=True
        )
        
        progress_bar.progress(100, text="✅ Ready!")
        progress_bar.empty()
        
        return {
            'hybrid_search': hybrid_search,
            'df': df,
            'collection': collection,
            'model': model,
            'folder_name': folder_name
        }
    
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# QUERY EXECUTION
# ============================================================================

def execute_query(query: str, hybrid_search_obj) -> Dict:
    """
    Execute query using OptimizedHybridSearch (SP8)
    
    Args:
        query: User's natural language query
        hybrid_search_obj: OptimizedHybridSearch instance
    
    Returns:
        Dict with answer and metadata
    """
    try:
        with st.spinner("🔍 Processing your query..."):
            result = hybrid_search_obj.search_with_router(query, top_k=10)
            return result
    
    except Exception as e:
        st.error(f"❌ Error executing query: {e}")
        import traceback
        traceback.print_exc()
        return None


def format_result(result: Dict) -> str:
    """
    Format search result for display in chat
    
    Args:
        result: Result from search_with_router
    
    Returns:
        Formatted string for display
    """
    if not result:
        return "❌ No result"
    
    output = []
    query_type = result.get('query_type', 'UNKNOWN')
    
    # Header
    output.append(f"**📊 Query Type:** {query_type}")
    
    # Route info
    route_info = result.get('route_info', {})
    if route_info:
        output.append("\n**🎯 Detected Information:**")
        
        if route_info.get('semantic_concept'):
            output.append(f"- 💡 Topic: {route_info['semantic_concept']}")
        
        if route_info.get('person'):
            role_emoji = {'SENDER': '📤', 'RECIPIENT': '📥'}.get(route_info.get('person_role'), '👤')
            output.append(f"- {role_emoji} Person: {route_info['person']} ({route_info.get('person_role', 'UNKNOWN')})")
        
        if route_info.get('year') or route_info.get('month'):
            filters = []
            if route_info.get('year'):
                filters.append(f"Year: {route_info['year']}")
            if route_info.get('month'):
                months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                filters.append(f"Month: {months[route_info['month']]}")
            output.append(f"- 📅 {', '.join(filters)}")
        
        if route_info.get('fuzzy_applied'):
            output.append(f"- 🔄 Fuzzy Match: '{route_info['original_query']}' → '{route_info['normalized_query']}'")
    
    # Handle different query types
    if query_type == 'SEMANTIC' or query_type == 'HYBRID':
        # Show vector search results
        results = result.get('results', [])
        
        if results:
            output.append(f"\n**🔎 Top Results ({len(results)}):**\n")
            
            for i, res in enumerate(results[:5], 1):  # Show top 5
                metadata = res.get('metadata', {})
                score = res.get('score', 0)
                
                # Score interpretation (lower is better for distance)
                similarity = max(0, 1 - score) if score < 1 else 1 - (score / 100)
                similarity_pct = similarity * 100
                
                output.append(f"**{i}. [{similarity_pct:.1f}% match]**")
                output.append(f"   📝 {res.get('message', '')}")  # Full message, no truncation
                output.append(f"   👤 From: {metadata.get('sender', 'Unknown')}")
                if metadata.get('date'):
                    output.append(f"   📅 Date: {metadata.get('date', '')}")
                output.append("")
            
            if len(results) > 5:
                output.append(f"... and {len(results) - 5} more results")
        else:
            output.append("\n❌ No matching messages found")
    
    elif query_type == 'ANALYTICAL':
        # Show pandas agent result
        answer = result.get('answer')
        code = result.get('code')
        
        if answer:
            output.append(f"\n**📊 Answer:**\n{answer}")
        
        if code and code != 'Error - no code generated':
            with st.expander("View Generated Code"):
                st.code(code, language='python')
    
    elif query_type == 'UNCLEAR':
        output.append("\n❓ I couldn't understand your query. Please try:")
        suggestions = result.get('suggestions', [
            "Add a topic: 'funny messages', 'birthday wishes'",
            "Specify a person: 'messages from Amit', 'wishes to Priya'",
            "Add a time filter: 'messages in 2024', 'last month'",
            "Use analytical terms: 'count messages', 'who sent most'"
        ])
        for sugg in suggestions:
            output.append(f"\n- {sugg}")
    
    return "\n".join(output)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Initialize session state
    initialize_session_state()
    
    # ========================================================================
    # SIDEBAR - CONFIGURATION
    # ========================================================================
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Folder selection
        folder_options = ['parivar', 'vellapanti', 'custom']
        folder_name = st.selectbox(
            "📁 Select Folder:",
            folder_options,
            help="Choose which WhatsApp group to analyze"
        )
        
        if folder_name == 'custom':
            folder_name = st.text_input("Enter folder name:", value="parivar")
        
        # File selection
        file_name = st.text_input(
            "📄 File name (without extension):",
            value=folder_name,
            help="Should match the _features.csv filename"
        )
        
        # Load button
        if st.button("🔄 Load Data", use_container_width=True, type="primary"):
            st.session_state.initialized = False
            st.rerun()
        
        st.divider()
        st.subheader("📖 Help")
        st.markdown("""
        **Query Examples:**
        - "messages from Rohit in 2026"
        - "funny messages from Priya in 2025"
        - "who is scared of ghosts"
        - "who sent the most messages"
        - "who is foodie"
        """)
    
    # ========================================================================
    # MAIN CONTENT - CHAT INTERFACE
    # ========================================================================
    
    st.title("💬 WhatsApp QA Chat")
    st.markdown("Ask questions about your WhatsApp group chat!")
    
    # Load data if not already loaded
    if not st.session_state.initialized:
        with st.spinner("Loading data..."):
            data_result = load_hybrid_search(folder_name, file_name)
            
            if data_result:
                st.session_state.hybrid_search = data_result['hybrid_search']
                st.session_state.df = data_result['df']
                st.session_state.collection = data_result['collection']
                st.session_state.model = data_result['model']
                st.session_state.folder_name = data_result['folder_name']
                st.session_state.initialized = True
            else:
                st.error("❌ Failed to load data. Check sidebar configuration.")
                st.stop()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])
    
    # Input field
    if prompt := st.chat_input("Ask me anything about the WhatsApp group...", key="chat_input"):
        
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": "👤"
        })
        
        # Display user message
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
        
        # Execute query
        result = execute_query(prompt, st.session_state.hybrid_search)
        
        # Format and display response
        if result:
            response_text = format_result(result)
            
            # Add bot message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "avatar": "🤖"
            })
            
            # Display bot response
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(response_text)
        else:
            error_msg = "❌ Failed to process your query. Please try again."
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "avatar": "🤖"
            })
            
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(error_msg)
    
    # Clear chat button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
