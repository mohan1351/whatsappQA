"""
Test SP5 Embedding Quality - Rigorous Evaluation
Tests SEMANTIC and HYBRID search quality with Precision & Recall metrics
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from smart_query_generator import SmartQueryGenerator


class EmbeddingQualityTester:
    """
    Rigorous embedding quality tester with precision and recall metrics
    """
    
    def __init__(self, folder_name: str, file_name: str):
        self.folder_name = folder_name
        self.file_name = file_name
        
        # Paths
        self.db_path = f"./chroma_db_{folder_name}"
        self.csv_path = f"./data/whatsapp/{folder_name}/{file_name}_features.csv"
        
        # Load resources
        print(f"{'='*80}")
        print(f"EMBEDDING QUALITY TEST: {folder_name}/{file_name}")
        print(f"{'='*80}")
        
        print("\n[1/4] Loading sentence transformer model...")
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("✅ Model loaded: all-mpnet-base-v2 (768 dimensions)")
        
        print("\n[2/4] Loading ChromaDB...")
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"ChromaDB not found at {self.db_path}. Run sp5 first.")
        
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(f"whatsapp_{folder_name}")
        print(f"✅ ChromaDB loaded: {self.collection.count()} messages")
        
        print("\n[3/4] Loading CSV data...")
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV not found at {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ CSV loaded: {len(self.df)} rows")
        
        print("\n[4/4] Analyzing data structure...")
        self._analyze_data()
        
        # Test results storage
        self.test_results = []
        
        # Smart query generator for name normalization
        print("\n[5/5] Loading Smart Query Generator...")
        try:
            self.smart_query_gen = SmartQueryGenerator(folder_name=folder_name)
            print("✅ Smart Query Generator loaded")
        except Exception as e:
            print(f"⚠️  Smart Query Generator not available: {e}")
            self.smart_query_gen = None
        
    def _analyze_data(self):
        """Analyze available metadata for test planning"""
        print(f"   Columns: {', '.join(self.df.columns.tolist())}")
        
        # Senders
        senders = self.df[self.df['sender'] != 'system']['sender'].value_counts()
        print(f"\n   Top Senders ({len(senders)} total):")
        for sender, count in senders.head(5).items():
            print(f"      - {sender}: {count} messages")
        
        # Birthday persons
        birthday_persons = self.df[self.df['birthday_person'].notna()]['birthday_person'].value_counts()
        if len(birthday_persons) > 0:
            print(f"\n   Birthday Persons ({len(birthday_persons)} total):")
            for person, count in birthday_persons.head(5).items():
                print(f"      - {person}: {count} birthday messages")
        
        # Years and months
        years = self.df['year'].value_counts().sort_index()
        months = self.df['month'].value_counts().sort_index()
        print(f"\n   Time Range: {years.index.min()}-{years.index.max()} | Months: {sorted(months.index.tolist())}")
        
        print(f"{'='*80}\n")
    
    def semantic_search(self, query: str, top_k: int = 5):
        """
        Pure semantic search (no metadata filters)
        """
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results
    
    def hybrid_search(self, query: str, metadata_filter: dict, top_k: int = 5):
        """
        Hybrid search (semantic + metadata filters)
        """
        query_embedding = self.model.encode(query).tolist()
        
        # Format metadata filter for ChromaDB
        # Single filter: {"key": "value"}
        # Multiple filters: {"$and": [{"key1": "value1"}, {"key2": "value2"}]}
        if len(metadata_filter) > 1:
            where_clause = {
                "$and": [{k: v} for k, v in metadata_filter.items()]
            }
        else:
            where_clause = metadata_filter
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )
        
        return results
    
    def calculate_precision_recall(self, retrieved_ids: list, ground_truth_ids: list, total_relevant: int):
        """
        Calculate Precision@K and Recall@K
        
        Precision@K = (Relevant in top K) / K
        Recall@K = (Relevant in top K) / Total Relevant in Dataset
        """
        retrieved_set = set(retrieved_ids)
        ground_truth_set = set(ground_truth_ids)
        
        # True Positives: Retrieved AND relevant
        true_positives = retrieved_set.intersection(ground_truth_set)
        
        # Precision: What fraction of retrieved items are relevant?
        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0.0
        
        # Recall: What fraction of relevant items did we retrieve?
        recall = len(true_positives) / total_relevant if total_relevant > 0 else 0.0
        
        return precision, recall, len(true_positives)
    
    def display_results(self, test_name: str, query: str, results: dict, 
                       search_type: str, metadata_filter: dict = None):
        """
        Display search results in readable format
        """
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print(f"{'='*80}")
        print(f"Query: '{query}'")
        print(f"Type: {search_type}")
        if metadata_filter:
            print(f"Filter: {metadata_filter}")
        print(f"{'-'*80}")
        
        ids = results['ids'][0]
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        # Convert distances to similarity scores (cosine similarity)
        similarities = [1 - d for d in distances]
        
        displayed_results = []
        
        for i, (doc_id, doc, sim, meta) in enumerate(zip(ids, documents, similarities, metadatas), 1):
            sender = meta.get('sender', 'Unknown')
            birthday_person = meta.get('birthday_person', '')
            year = meta.get('year', '')
            month = meta.get('month', '')
            
            # Truncate message for display
            display_msg = doc[:80] + "..." if len(doc) > 80 else doc
            
            print(f"\nResult {i} | Score: {sim:.4f}")
            print(f"   Message: {display_msg}")
            print(f"   Sender: {sender} | To: {birthday_person if birthday_person else 'N/A'} | Date: {year}-{month:02d}" if year else f"   Sender: {sender}")
            
            displayed_results.append({
                'id': doc_id,
                'document': doc,
                'similarity': sim,
                'sender': sender,
                'birthday_person': birthday_person
            })
        
        avg_score = np.mean(similarities) if similarities else 0.0
        min_score = min(similarities) if similarities else 0.0
        max_score = max(similarities) if similarities else 0.0
        
        print(f"\n{'-'*80}")
        print(f"Score Stats: Avg={avg_score:.4f}, Min={min_score:.4f}, Max={max_score:.4f}")
        
        return displayed_results, similarities
    
    def test_semantic_search(self, test_name: str, query: str, 
                            expected_keywords: list = None, 
                            ground_truth_filter: dict = None,
                            top_k: int = 5):
        """
        Test semantic search quality
        
        Args:
            test_name: Name of the test
            query: Search query
            expected_keywords: Keywords that should appear in results (for manual check)
            ground_truth_filter: Dict to filter ground truth from CSV (e.g., {'sender': 'Amit'})
            top_k: Number of results to retrieve
        """
        print(f"\n{'#'*80}")
        print(f"SEMANTIC SEARCH TEST: {test_name}")
        print(f"{'#'*80}")
        
        # Run search
        results = self.semantic_search(query, top_k=top_k)
        
        # Display results
        displayed, similarities = self.display_results(
            test_name, query, results, "SEMANTIC"
        )
        
        # Calculate ground truth for precision/recall
        retrieved_ids = results['ids'][0]
        
        precision, recall, true_positives = None, None, None
        
        if ground_truth_filter:
            # Calculate ground truth from CSV
            ground_truth_df = self.df.copy()
            for key, value in ground_truth_filter.items():
                if key == 'birthday_person':
                    ground_truth_df = ground_truth_df[ground_truth_df[key].notna()]
                    ground_truth_df = ground_truth_df[ground_truth_df[key] == value]
                else:
                    ground_truth_df = ground_truth_df[ground_truth_df[key] == value]
            
            total_relevant = len(ground_truth_df)
            ground_truth_ids = [f"msg_{idx}" for idx in ground_truth_df.index.tolist()]
            
            precision, recall, true_positives = self.calculate_precision_recall(
                retrieved_ids, ground_truth_ids, total_relevant
            )
            
            print(f"\n{'='*80}")
            print(f"PRECISION & RECALL (Ground Truth: {ground_truth_filter})")
            print(f"{'='*80}")
            print(f"Total Relevant Messages in Dataset: {total_relevant}")
            print(f"Retrieved: {len(retrieved_ids)}")
            print(f"True Positives (Relevant & Retrieved): {true_positives}")
            print(f"\nPrecision@{top_k}: {precision:.4f} ({true_positives}/{len(retrieved_ids)})")
            print(f"Recall@{top_k}: {recall:.4f} ({true_positives}/{total_relevant})")
            
            if precision == 1.0:
                print("✅ Perfect Precision - All retrieved results are relevant!")
            elif precision >= 0.8:
                print("✅ High Precision - Most results are relevant")
            elif precision >= 0.6:
                print("⚠️  Moderate Precision - Some irrelevant results")
            else:
                print("❌ Low Precision - Many irrelevant results")
        
        # Store results
        self.test_results.append({
            'test_name': test_name,
            'query': query,
            'search_type': 'SEMANTIC',
            'avg_similarity': np.mean(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'retrieved': len(retrieved_ids),
            'expected_keywords': expected_keywords
        })
        
        return displayed, similarities
    
    def test_hybrid_search(self, test_name: str, query: str, 
                          metadata_filter: dict,
                          expected_sender: str = None,
                          expected_birthday_person: str = None,
                          top_k: int = 5):
        """
        Test hybrid search quality (semantic + metadata)
        
        Args:
            test_name: Name of the test
            query: Search query
            metadata_filter: Metadata filters (e.g., {'sender': 'Amit'})
            expected_sender: Expected sender in results (for validation)
            expected_birthday_person: Expected birthday person (for validation)
            top_k: Number of results to retrieve
        """
        print(f"\n{'#'*80}")
        print(f"HYBRID SEARCH TEST: {test_name}")
        print(f"{'#'*80}")
        
        # Run search
        results = self.hybrid_search(query, metadata_filter, top_k=top_k)
        
        # Display results
        displayed, similarities = self.display_results(
            test_name, query, results, "HYBRID", metadata_filter
        )
        
        # Validate metadata filtering
        retrieved_ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        
        # Check filter effectiveness
        filter_accuracy = 100.0
        filter_errors = []
        
        for meta in metadatas:
            for key, expected_value in metadata_filter.items():
                actual_value = meta.get(key, '')
                if actual_value != expected_value:
                    filter_errors.append(f"{key}: expected '{expected_value}', got '{actual_value}'")
                    filter_accuracy = 0.0
        
        print(f"\n{'='*80}")
        print(f"METADATA FILTER ACCURACY")
        print(f"{'='*80}")
        if filter_accuracy == 100.0:
            print(f"✅ 100% Filter Accuracy - All results match metadata filters")
        else:
            print(f"❌ Filter Failed - Errors found:")
            for error in filter_errors:
                print(f"   - {error}")
        
        # Calculate precision/recall based on metadata filter
        ground_truth_df = self.df.copy()
        for key, value in metadata_filter.items():
            if key == 'birthday_person' and value:
                ground_truth_df = ground_truth_df[ground_truth_df[key].notna()]
                ground_truth_df = ground_truth_df[ground_truth_df[key] == value]
            else:
                ground_truth_df = ground_truth_df[ground_truth_df[key] == value]
        
        total_relevant = len(ground_truth_df)
        ground_truth_ids = [f"msg_{idx}" for idx in ground_truth_df.index.tolist()]
        
        precision, recall, true_positives = self.calculate_precision_recall(
            retrieved_ids, ground_truth_ids, total_relevant
        )
        
        print(f"\nTotal Messages Matching Filter: {total_relevant}")
        print(f"Retrieved: {len(retrieved_ids)}")
        print(f"True Positives: {true_positives}")
        print(f"\nPrecision@{top_k}: {precision:.4f}")
        print(f"Recall@{top_k}: {recall:.4f}")
        
        # Store results
        self.test_results.append({
            'test_name': test_name,
            'query': query,
            'search_type': 'HYBRID',
            'metadata_filter': str(metadata_filter),
            'avg_similarity': np.mean(similarities) if similarities else 0.0,
            'min_similarity': min(similarities) if similarities else 0.0,
            'max_similarity': max(similarities) if similarities else 0.0,
            'filter_accuracy': filter_accuracy,
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'retrieved': len(retrieved_ids),
            'total_relevant': total_relevant
        })
        
        return displayed, similarities
    
    def run_test_suite(self):
        """
        Run comprehensive test suite
        """
        print(f"\n{'#'*80}")
        print(f"STARTING COMPREHENSIVE TEST SUITE")
        print(f"{'#'*80}")
        
        # Get sample data for tests
        sample_senders = self.df[self.df['sender'] != 'system']['sender'].value_counts()
        top_sender = sample_senders.index[0] if len(sample_senders) > 0 else None
        
        sample_birthday = self.df[self.df['birthday_person'].notna()]['birthday_person'].value_counts()
        top_birthday_person = sample_birthday.index[0] if len(sample_birthday) > 0 else None
        
        sample_years = self.df['year'].value_counts().index.tolist()
        sample_year = sample_years[0] if sample_years else 2024
        
        print(f"\nTest Parameters:")
        print(f"   Top Sender: {top_sender}")
        print(f"   Top Birthday Person: {top_birthday_person}")
        print(f"   Sample Year: {sample_year}")
        
        # ========================================================================
        # SEMANTIC TESTS (No Metadata Filters)
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("PART 1: SEMANTIC SEARCH TESTS (No Metadata)")
        print(f"{'='*80}")
        
        self.test_semantic_search(
            test_name="Generic Birthday Wishes",
            query="birthday wishes happy birthday celebrations",
            ground_truth_filter={'birthday_person': top_birthday_person} if top_birthday_person else None,
            top_k=5
        )
        
        self.test_semantic_search(
            test_name="Funny Messages",
            query="funny jokes humor laughter comedy",
            expected_keywords=['funny', 'joke', 'laugh', '😂', 'lol'],
            top_k=5
        )
        
        self.test_semantic_search(
            test_name="Greetings",
            query="good morning good night hello hi greetings",
            expected_keywords=['morning', 'night', 'hello', 'hi', 'hey'],
            top_k=5
        )
        
        self.test_semantic_search(
            test_name="Travel Plans",
            query="flight airport taxi travel journey trip",
            expected_keywords=['flight', 'airport', 'travel', 'taxi', 'journey'],
            top_k=5
        )
        
        self.test_semantic_search(
            test_name="Food Discussions",
            query="dinner lunch breakfast food restaurant eating",
            expected_keywords=['dinner', 'lunch', 'food', 'eat', 'restaurant'],
            top_k=5
        )
        
        # ========================================================================
        # HYBRID TESTS (Semantic + Metadata)
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("PART 2: HYBRID SEARCH TESTS (Semantic + Metadata)")
        print(f"{'='*80}")
        
        if top_sender:
            self.test_hybrid_search(
                test_name=f"Messages from {top_sender}",
                query="messages conversations chat",
                metadata_filter={'sender': top_sender},
                expected_sender=top_sender,
                top_k=5
            )
        
        if top_birthday_person:
            self.test_hybrid_search(
                test_name=f"Birthday wishes to {top_birthday_person}",
                query="birthday wishes happy birthday celebrations",
                metadata_filter={'birthday_person': top_birthday_person},
                expected_birthday_person=top_birthday_person,
                top_k=5
            )
        
        self.test_hybrid_search(
            test_name=f"Messages in {sample_year}",
            query="messages conversations",
            metadata_filter={'year': sample_year},
            top_k=5
        )
        
        # Combined filters
        if top_sender:
            self.test_hybrid_search(
                test_name=f"Messages from {top_sender} in {sample_year}",
                query="conversations chat messages",
                metadata_filter={'sender': top_sender, 'year': sample_year},
                expected_sender=top_sender,
                top_k=5
            )
        
        # ========================================================================
        # SUMMARY
        # ========================================================================
        
        self.print_summary()
    
    def print_summary(self):
        """
        Print comprehensive test summary
        """
        print(f"\n{'#'*80}")
        print(f"TEST SUITE SUMMARY")
        print(f"{'#'*80}")
        
        df_results = pd.DataFrame(self.test_results)
        
        # Overall statistics
        print(f"\nTotal Tests Run: {len(df_results)}")
        print(f"   Semantic Tests: {len(df_results[df_results['search_type'] == 'SEMANTIC'])}")
        print(f"   Hybrid Tests: {len(df_results[df_results['search_type'] == 'HYBRID'])}")
        
        # Similarity score statistics
        print(f"\n{'='*80}")
        print("SIMILARITY SCORES")
        print(f"{'='*80}")
        print(f"Average Similarity (All): {df_results['avg_similarity'].mean():.4f}")
        print(f"Average Similarity (Semantic): {df_results[df_results['search_type'] == 'SEMANTIC']['avg_similarity'].mean():.4f}")
        print(f"Average Similarity (Hybrid): {df_results[df_results['search_type'] == 'HYBRID']['avg_similarity'].mean():.4f}")
        
        # Precision & Recall
        print(f"\n{'='*80}")
        print("PRECISION & RECALL")
        print(f"{'='*80}")
        
        # Filter out None values
        precision_data = df_results[df_results['precision'].notna()]
        recall_data = df_results[df_results['recall'].notna()]
        
        if len(precision_data) > 0:
            print(f"Average Precision@5: {precision_data['precision'].mean():.4f}")
            print(f"Average Recall@5: {recall_data['recall'].mean():.4f}")
            
            # Quality assessment
            avg_precision = precision_data['precision'].mean()
            avg_recall = recall_data['recall'].mean()
            
            print(f"\n{'='*80}")
            print("QUALITY ASSESSMENT")
            print(f"{'='*80}")
            
            if avg_precision >= 0.8:
                print("✅ EXCELLENT Precision - Highly relevant results")
            elif avg_precision >= 0.6:
                print("✅ GOOD Precision - Mostly relevant results")
            elif avg_precision >= 0.4:
                print("⚠️  MODERATE Precision - Some irrelevant results")
            else:
                print("❌ POOR Precision - Many irrelevant results")
            
            if avg_recall >= 0.5:
                print("✅ GOOD Recall - Finding most relevant messages")
            elif avg_recall >= 0.3:
                print("⚠️  MODERATE Recall - Missing some relevant messages")
            else:
                print("❌ LOW Recall - Missing many relevant messages")
        
        # Filter accuracy (for hybrid searches)
        hybrid_results = df_results[df_results['search_type'] == 'HYBRID']
        if len(hybrid_results) > 0:
            avg_filter_accuracy = hybrid_results['filter_accuracy'].mean()
            print(f"\n{'='*80}")
            print("METADATA FILTER ACCURACY")
            print(f"{'='*80}")
            print(f"Average Filter Accuracy: {avg_filter_accuracy:.1f}%")
            if avg_filter_accuracy == 100.0:
                print("✅ Perfect metadata filtering!")
            else:
                print("❌ Metadata filters not working correctly")
        
        # Detailed results table
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print(f"{'='*80}")
        
        display_cols = ['test_name', 'search_type', 'avg_similarity', 'precision', 'recall']
        available_cols = [col for col in display_cols if col in df_results.columns]
        print(df_results[available_cols].to_string(index=False))
        
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        avg_sim = df_results['avg_similarity'].mean()
        if avg_sim >= 0.7:
            print("✅ Embedding quality is EXCELLENT")
            print("   → High similarity scores indicate strong semantic understanding")
        elif avg_sim >= 0.5:
            print("✅ Embedding quality is GOOD")
            print("   → Reasonable similarity scores for RAG applications")
        else:
            print("⚠️  Embedding quality may need improvement")
            print("   → Consider:")
            print("      - Better message preprocessing")
            print("      - Different embedding model")
            print("      - More training data")
        
        print(f"\n{'='*80}")
        print(f"Test completed for: {self.folder_name}/{self.file_name}")
        print(f"ChromaDB: {self.db_path}")
        print(f"CSV: {self.csv_path}")
        print(f"{'='*80}\n")
    
    def interactive_mode(self):
        """
        Interactive mode - enter your own queries and test in real-time
        """
        print(f"\n{'#'*80}")
        print(f"INTERACTIVE TESTING MODE")
        print(f"{'#'*80}")
        print(f"\nDatabase: {self.folder_name}/{self.file_name}")
        print(f"Total messages: {self.collection.count()}")
        print(f"\nEnter your queries to test embedding quality in real-time!")
        print(f"Type 'quit' or 'exit' to stop.\n")
        
        # Show available senders and birthday persons for reference
        sample_senders = self.df[self.df['sender'] != 'system']['sender'].value_counts()
        birthday_persons = self.df[self.df['birthday_person'].notna()]['birthday_person'].value_counts()
        
        print(f"{'='*80}")
        print("AVAILABLE FILTERS (for reference)")
        print(f"{'='*80}")
        print(f"Senders: {', '.join(sample_senders.head(10).index.tolist())}")
        if len(birthday_persons) > 0:
            print(f"Birthday Persons: {', '.join(birthday_persons.head(10).index.tolist())}")
        print(f"Years: {', '.join(map(str, sorted(self.df['year'].unique())))}")
        print(f"{'='*80}\n")
        
        while True:
            try:
                # Get query
                print(f"\n{'='*80}")
                query = input("Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Exiting interactive mode. Goodbye!")
                    break
                
                if not query:
                    print("⚠️  Empty query. Please enter something.")
                    continue
                
                # Choose search type
                print("\nSearch type:")
                print("  1. SEMANTIC (no filters)")
                print("  2. HYBRID (with metadata filters)")
                print("  3. SMART (auto-normalize if score < 0.8)")
                search_choice = input("Choose (1, 2 or 3): ").strip()
                
                if search_choice not in ['1', '2', '3']:
                    print("⚠️  Invalid choice. Using SEMANTIC search.")
                    search_choice = '1'
                
                # Get top_k
                top_k_input = input("Number of results to show (default 5): ").strip()
                top_k = int(top_k_input) if top_k_input.isdigit() else 5
                
                # SEMANTIC SEARCH
                if search_choice == '1':
                    print(f"\n{'='*80}")
                    print(f"SEMANTIC SEARCH: '{query}'")
                    print(f"{'='*80}")
                    
                    results = self.semantic_search(query, top_k=top_k)
                    self._display_interactive_results(results, query, "SEMANTIC")
                
                # SMART SEARCH (auto-normalize if score < 0.8)
                elif search_choice == '3':
                    self._smart_search_interactive(query, top_k=top_k)
                
                # HYBRID SEARCH
                else:
                    metadata_filter = {}
                    
                    # Collect filters
                    print("\nEnter metadata filters (press Enter to skip):")
                    
                    sender = input("  Sender name: ").strip()
                    if sender:
                        metadata_filter['sender'] = sender
                    
                    birthday_person = input("  Birthday person: ").strip()
                    if birthday_person:
                        metadata_filter['birthday_person'] = birthday_person
                    
                    year = input("  Year (e.g., 2024): ").strip()
                    if year.isdigit():
                        metadata_filter['year'] = int(year)
                    
                    month = input("  Month (1-12): ").strip()
                    if month.isdigit():
                        metadata_filter['month'] = int(month)
                    
                    if not metadata_filter:
                        print("⚠️  No filters provided. Switching to SEMANTIC search.")
                        results = self.semantic_search(query, top_k=top_k)
                        self._display_interactive_results(results, query, "SEMANTIC")
                    else:
                        print(f"\n{'='*80}")
                        print(f"HYBRID SEARCH: '{query}'")
                        print(f"Filters: {metadata_filter}")
                        print(f"{'='*80}")
                        
                        results = self.hybrid_search(query, metadata_filter, top_k=top_k)
                        self._display_interactive_results(results, query, "HYBRID", metadata_filter)
                
                # Ask to continue
                print(f"\n{'='*80}")
                continue_choice = input("\nTest another query? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '']:
                    print("\n👋 Exiting interactive mode. Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Exiting interactive mode.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")
    
    def _display_interactive_results(self, results: dict, query: str, 
                                     search_type: str, metadata_filter: dict = None):
        """
        Display results for interactive mode
        """
        ids = results['ids'][0]
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        if not ids:
            print("\n❌ No results found!")
            if metadata_filter:
                print("   Try different filters or a broader query.")
            return
        
        # Convert distances to similarity scores
        similarities = [1 - d for d in distances]
        
        print(f"\n{'='*80}")
        print(f"FOUND {len(ids)} RESULTS")
        print(f"{'='*80}")
        
        for i, (doc_id, doc, sim, meta) in enumerate(zip(ids, documents, similarities, metadatas), 1):
            sender = meta.get('sender', 'Unknown')
            birthday_person = meta.get('birthday_person', '')
            year = meta.get('year', '')
            month = meta.get('month', '')
            
            # Truncate for display
            display_msg = doc[:120] + "..." if len(doc) > 120 else doc
            
            print(f"\n{'─'*80}")
            print(f"Result #{i} | Similarity: {sim:.4f} {'🔥🔥🔥' if sim > 0.8 else '🔥' if sim > 0.6 else ''}")
            print(f"{'─'*80}")
            print(f"Message: {display_msg}")
            print(f"Sender: {sender}", end='')
            if birthday_person:
                print(f" | To: {birthday_person}", end='')
            if year:
                print(f" | Date: {year}-{month:02d}", end='')
            print()
        
        # Summary stats
        avg_score = np.mean(similarities)
        print(f"\n{'='*80}")
        print(f"QUALITY METRICS")
        print(f"{'='*80}")
        print(f"Average Similarity: {avg_score:.4f}")
        print(f"Best Match: {max(similarities):.4f}")
        print(f"Worst Match: {min(similarities):.4f}")
        
        if avg_score >= 0.7:
            print("✅ Excellent match quality!")
        elif avg_score >= 0.5:
            print("✅ Good match quality")
        else:
            print("⚠️  Low match quality - consider refining your query")
        
        # Metadata validation for hybrid
        if metadata_filter:
            print(f"\n{'='*80}")
            print(f"FILTER VALIDATION")
            print(f"{'='*80}")
            all_match = True
            for meta in metadatas:
                for key, expected in metadata_filter.items():
                    actual = meta.get(key, '')
                    if actual != expected:
                        print(f"❌ Filter mismatch: {key} = '{actual}' (expected '{expected}')")
                        all_match = False
            
            if all_match:
                print("✅ All results match the filters perfectly!")
    
    def _smart_search_interactive(self, query: str, top_k: int = 5):
        """
        Smart search: runs query1 first, if best score < 0.8, also runs query2 (normalized)
        Returns the better result.
        """
        query1 = query  # Original query
        
        print(f"\n{'='*80}")
        print(f"SMART SEARCH: '{query1}'")
        print(f"{'='*80}")
        
        # Step 1: Search with original query (query1)
        print(f"\n[Step 1] Searching with query1: '{query1}'")
        results1 = self.semantic_search(query1, top_k=top_k)
        
        # Get best score from query1
        if results1['distances'][0]:
            best_score1 = 1 - min(results1['distances'][0])  # Convert distance to similarity
        else:
            best_score1 = 0.0
        
        print(f"   Best similarity score: {best_score1:.4f}")
        
        # Step 2: Check if we need to try query2
        if best_score1 >= 0.8:
            print(f"   ✅ Score >= 0.8, using query1 results")
            self._display_interactive_results(results1, query1, "SMART (query1)")
            return
        
        # Score < 0.8, try to normalize
        print(f"   ⚠️  Score < 0.8, trying normalized query...")
        
        if not self.smart_query_gen:
            print(f"   ❌ Smart Query Generator not available, using query1 results")
            self._display_interactive_results(results1, query1, "SMART (query1)")
            return
        
        # Generate query2 (normalized)
        query2, info = self.smart_query_gen.generate(query1)
        
        if not info['normalized']:
            print(f"   ⚠️  No normalization possible: {info['reason']}")
            print(f"   Using query1 results")
            self._display_interactive_results(results1, query1, "SMART (query1)")
            return
        
        # Step 3: Search with normalized query (query2)
        print(f"\n[Step 2] Searching with query2: '{query2}'")
        for r in info['replacements']:
            print(f"   → '{r['original']}' normalized to '{r['canonical']}'")
        
        results2 = self.semantic_search(query2, top_k=top_k)
        
        if results2['distances'][0]:
            best_score2 = 1 - min(results2['distances'][0])
        else:
            best_score2 = 0.0
        
        print(f"   Best similarity score: {best_score2:.4f}")
        
        # Step 4: Compare and pick winner
        print(f"\n{'='*80}")
        print(f"COMPARISON")
        print(f"{'='*80}")
        print(f"query1 (original):   '{query1}' → Best Score: {best_score1:.4f}")
        print(f"query2 (normalized): '{query2}' → Best Score: {best_score2:.4f}")
        
        if best_score2 > best_score1:
            print(f"\n🏆 WINNER: query2 (normalized) with +{best_score2 - best_score1:.4f} improvement")
            self._display_interactive_results(results2, query2, "SMART (query2 - normalized)")
        else:
            print(f"\n🏆 WINNER: query1 (original)")
            self._display_interactive_results(results1, query1, "SMART (query1 - original)")


def main():
    """Main function to run tests"""
    parser = argparse.ArgumentParser(description='Test SP5 Embedding Quality')
    parser.add_argument(
        'folder_name',
        type=str,
        help='Folder name (e.g., parivar, vellapanti)'
    )
    parser.add_argument(
        'file_name',
        type=str,
        help='File name without extension (e.g., parivar, vellapanti)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['auto', 'interactive', 'both'],
        default='auto',
        help='Test mode: auto (predefined tests), interactive (enter queries), both (run both modes)'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = EmbeddingQualityTester(args.folder_name, args.file_name)
    
    # Run based on mode
    if args.mode == 'auto':
        tester.run_test_suite()
    elif args.mode == 'interactive':
        tester.interactive_mode()
    elif args.mode == 'both':
        tester.run_test_suite()
        print("\n" + "="*80)
        print("Switching to INTERACTIVE MODE...")
        print("="*80)
        tester.interactive_mode()


if __name__ == "__main__":
    main()
