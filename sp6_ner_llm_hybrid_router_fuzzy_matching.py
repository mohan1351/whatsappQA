"""
NER + LLM Hybrid Router
Combines spaCy NER for entity extraction with LLM for semantic concept detection

Routing Logic:
1. IF NER finds entities AND LLM finds NO semantic → ANALYTICAL
2. ELIF LLM finds semantic AND NER finds NO entities → SEMANTIC  
3. ELIF both find something → HYBRID
4. ELSE neither found → UNCLEAR (ask user for clarification)

Advantages:
- NER: Fast, accurate entity extraction (persons, dates, numbers)
- LLM: Smart semantic concept detection with full context
- Person role detection: SENDER vs RECIPIENT
"""

import spacy
import json
import ollama
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import os
from dotenv import load_dotenv
from smart_query_generator import SmartQueryGenerator

# Load environment variables from .env file
load_dotenv()


class NERLLMHybridRouter:
    """
    Hybrid router combining NER entity extraction with LLM semantic classification
    Supports OpenAI (production) or Ollama (free/local)
    """
    
    def __init__(self, 
                 folder_name: str,
                 provider: str = "openai",
                 model: str = None,
                 api_key: str = None,
                 fuzzy_threshold: float = 0.75):
        """
        Initialize router with spaCy, LLM (OpenAI or Ollama), and Fuzzy Matching
        
        Args:
            folder_name: Folder name for name_pairs (e.g., 'parivar', 'vellapanti')
            provider: "openai" (recommended for production) or "ollama" (free, local)
            model: Model name
                  - OpenAI: "gpt-4o-mini" (default, best), "gpt-3.5-turbo" (cheaper)
                  - Ollama: "deepseek-r1:8b", "qwen2.5:7b", "deepseek-coder:6.7b"
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            fuzzy_threshold: Similarity threshold for fuzzy name matching (default 0.75)
        """
        self.folder_name = folder_name
        self.provider = provider.lower()
        
        # Initialize Smart Query Generator for fuzzy matching
        print("\n🔍 Initializing Fuzzy Name Matching...")
        try:
            self.smart_query_gen = SmartQueryGenerator(
                folder_name=folder_name, 
                threshold=fuzzy_threshold
            )
            print("✅ Fuzzy Name Matching enabled")
        except Exception as e:
            print(f"⚠️  Fuzzy matching not available: {e}")
            self.smart_query_gen = None
        
        # Setup OpenAI
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
                self.model = model or "gpt-4o-mini"
                print(f"🤖 Using OpenAI: {self.model} (PRODUCTION - Reliable routing)")
            except Exception as e:
                print(f"❌ OpenAI setup failed: {e}")
                print("   Falling back to Ollama...")
                self.provider = "ollama"
                self.model = model or "deepseek-coder:6.7b"
        
        # Setup Ollama
        if self.provider == "ollama":
            self.model = model or "deepseek-coder:6.7b"
            print(f"🤖 Using Ollama: {self.model} (FREE - Local)")
            
            # Test Ollama connection
            try:
                ollama.generate(model=self.model, prompt="test", options={'num_predict': 1})
                print(f"✅ Ollama connected")
            except Exception as e:
                print(f"⚠️  Ollama connection failed: {e}")
        
        # Load spaCy NER model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy NER model loaded")
        except OSError:
            print("⚠️  Installing spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model installed and loaded")
        
        print("✅ NER + LLM Hybrid Router initialized")
    
    def extract_entities_with_ner(self, query: str) -> Dict[str, any]:
        """
        Extract entities using spaCy NER
        
        Args:
            query: Natural language query
        
        Returns:
            Dict with entities, temporal filters, and flags
        """
        doc = self.nlp(query)
        
        entities = {
            'persons': [],
            'dates': [],
            'numbers': [],
            'organizations': [],
            'locations': []
        }
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['persons'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ in ['CARDINAL', 'QUANTITY']:
                entities['numbers'].append(ent.text)
            elif ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'GPE':
                entities['locations'].append(ent.text)
        
        # Parse temporal expressions
        temporal_filters = {}
        for date_str in entities['dates']:
            temporal_filters.update(self._parse_temporal_expression(date_str))
        
        # Check if any entities found
        has_entities = bool(
            entities['persons'] or 
            entities['dates'] or 
            entities['numbers']
        )
        
        return {
            'entities': entities,
            'temporal_filters': temporal_filters,
            'has_entities': has_entities
        }
    
    def _parse_temporal_expression(self, date_str: str) -> Dict[str, int]:
        """
        Parse date strings to year/month
        
        Args:
            date_str: Date expression
        
        Returns:
            Dict with year, month
        """
        temporal = {}
        
        # Extract year (2024, 2023, etc.)
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        if year_match:
            temporal['year'] = int(year_match.group(1))
        
        # Extract month name
        month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        date_lower = date_str.lower()
        for month_name, month_num in month_names.items():
            if month_name in date_lower:
                temporal['month'] = month_num
                break
        
        # Handle relative dates
        current_date = datetime.now()
        if 'last month' in date_lower:
            month = current_date.month - 1
            temporal['month'] = month if month > 0 else 12
        
        if 'this month' in date_lower or 'current month' in date_lower:
            temporal['month'] = current_date.month
        
        return temporal
    
    def _fallback_semantic_detection(self, query: str) -> str:
        """
        Fallback keyword-based semantic concept detection
        If LLM fails to detect semantic concept, check for common keywords
        
        Args:
            query: Query text
        
        Returns:
            Semantic concept or empty string
        """
        query_lower = query.lower()
        
        # Analytical keywords - NO semantic concept
        analytical_keywords = [
            'count', 'how many', 'total', 'number of', 'most', 'least',
            'top', 'bottom', 'average', 'sum', 'who sent', 'list all',
            'show all', 'group by', 'statistics'
        ]
        
        # Check if it's analytical
        for keyword in analytical_keywords:
            if keyword in query_lower:
                return ''  # No semantic concept
        
        # Semantic keywords - HAS semantic concept
        semantic_patterns = [
            'birthday', 'wishes', 'congratulations', 'congrats',
            'funny', 'joke', 'humor', 'philosophical', 'advice',
            'good morning', 'good night', 'thank', 'sorry',
            'celebration', 'festival', 'diwali', 'holi', 'christmas',
            'love', 'miss', 'care', 'worried', 'happy', 'sad',
            'excited', 'proud', 'achievement', 'success',
            # Attributes and preferences
            'favourite', 'favorite', 'hobby', 'likes', 'loves', 'enjoys',
            'food', 'dish', 'cuisine', 'sport', 'game', 'music', 'movie',
            'lives in', 'from', 'works as', 'job', 'profession', 'engineer'
        ]
        
        # Check if any semantic keyword present
        for pattern in semantic_patterns:
            if pattern in query_lower:
                # Return the original query as semantic concept
                return query.strip()
        
        return ''  # Default: no semantic concept
    
    def detect_semantic_with_llm(self, query: str) -> Dict[str, any]:
        """
        Use LLM to detect semantic concept and person role
        
        Args:
            query: Full natural language query
        
        Returns:
            Dict with semantic_concept, person, person_role, has_semantic
        """
        prompt = f"""Analyze this query for WhatsApp chat analysis.

Query: "{query}"

Extract:
1. semantic_concept: Is there a SEMANTIC CONCEPT (topic/theme/emotion/sentiment/attribute/preference)?
   - Topics: "funny messages", "birthday wishes", "philosophical thoughts", "holi messages"
   - Attributes/Preferences: "favourite food is rice", "hobby is cricket", "lives in Mumbai"
   - Descriptions: "tall person", "works as engineer", "loves traveling"
   - Empty string ONLY if purely analytical (counting, statistics, grouping)

2. person: Person name if mentioned (empty string if none)

3. person_role: Determine from context (VERY IMPORTANT - Must choose ONE):
   - "RECIPIENT": "to X", "X received", "X got", "wishes for X", "birthday wishes to X", "messages to X"
   - "SENDER": "from X", "X sent", "X wrote", "messages by X", "by X"
   - null: no person mentioned
   
   IMPORTANT: NEVER use "BOTH". Default to "SENDER" if direction is ambiguous.

EXAMPLES:
- "birthday wishes to Amit" → person="Amit", person_role="RECIPIENT"
- "messages from Priya" → person="Priya", person_role="SENDER"
- "who wished Amit" → person="Amit", person_role="RECIPIENT"
- "funny messages by Rohit" → person="Rohit", person_role="SENDER"

CRITICAL: Distinguish semantic concepts from analytical operations:
- "funny messages" → semantic concept ✅
- "birthday wishes" → semantic concept ✅
- "holi messages" → semantic concept ✅
- "favourite food is rice" → semantic concept ✅ (attribute/preference)
- "hobby is cricket" → semantic concept ✅ (attribute/preference)
- "lives in Mumbai" → semantic concept ✅ (attribute/location)
- "count messages" → NO semantic concept ❌ (analytical)
- "most messages" → NO semantic concept ❌ (analytical)
- "who sent most messages" → NO semantic concept ❌ (analytical - aggregation)

Return ONLY valid JSON:
{{
    
  "semantic_concept": "",
  "person": "",
  "person_role": "SENDER|RECIPIENT|null"
}}"""

        try:
            # Call LLM based on provider
            if self.provider == "openai":
                # OpenAI API call
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200
                )
                response_text = response.choices[0].message.content.strip()
            
            else:
                # Ollama API call
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    options={'temperature': 0, 'num_predict': 200}
                )
                response_text = response['response'].strip()
            
            # Parse JSON response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = {'semantic_concept': '', 'person': '', 'person_role': None}
            
            # Fallback: If LLM didn't detect semantic concept, check for common keywords
            semantic_concept = result.get('semantic_concept', '').strip()
            if not semantic_concept:
                semantic_concept = self._fallback_semantic_detection(query)
            
            # Determine if semantic concept exists
            has_semantic = bool(semantic_concept)
            
            return {
                'semantic_concept': semantic_concept,
                'person': result.get('person', '').strip() or None,
                'person_role': result.get('person_role') if result.get('person_role') != 'null' else None,
                'has_semantic': has_semantic
            }
            
        except Exception as e:
            print(f"⚠️  LLM call failed: {e}")
            return {
                'semantic_concept': None,
                'person': None,
                'person_role': None,
                'has_semantic': False
            }
    
    def detect_and_convert_negation(self, query: str) -> Tuple[bool, str]:
        """
        Use LLM to detect negation intent and convert to positive form
        
        Args:
            query: Original query
        
        Returns:
            Tuple of (is_negation, converted_query)
            - is_negation: True if query contains negation (finding what did NOT happen)
            - converted_query: Positive form of query (or original if no negation)
        """
        # Use LLM to detect AND convert in single call
        prompt = f"""Analyze if this query requires negation (finding what did NOT happen or did NOT match).

Query: "{query}"

NEGATION queries (is_negation=true) - MUST contain explicit negation words:
- "who did NOT wish Krati" → has "did NOT" (negation)
- "who didn't wish Krati" → has "didn't" (negation)
- "messages not from Amit" → has "not from" (exclusion)
- "everyone except Priya" → has "except" (exclusion)
- "who never replied" → has "never" (negation)
- "excluding January" → has "excluding" (exclusion)
- "without birthdays" → has "without" (exclusion)

NOT NEGATION (is_negation=false):
- "who did wish Krati" → positive query (auxiliary "did" is NOT negation)
- "did anyone wish Krati" → positive question (auxiliary "did" is NOT negation)
- "who wished Krati" → positive query
- "not sure which messages" → user uncertainty (not data negation)
- "messages about not giving up" → "not" is topic content (not query negation)
- "I'm not looking for birthdays" → user intent clarification (not data negation)

KEY RULES:
1. "did NOT" or "didn't" = NEGATION ✅
2. "did" alone (auxiliary verb) = NOT NEGATION ❌
3. "not", "never", "except", "excluding", "without" in query logic = NEGATION ✅
4. "not" in topic/content/uncertainty = NOT NEGATION ❌

Key distinction: Negation queries want to EXCLUDE/FIND-ABSENCE in data. Non-negation uses "not" for other purposes or uses "did" as auxiliary.

Return ONLY valid JSON (no markdown, no code blocks):
{{
  "is_negation": true or false,
  "converted_query": "positive form of query if negation, else original query"
}}

Examples:
- Query: "who did not wish Krati" → {{"is_negation": true, "converted_query": "who wished Krati"}}
- Query: "not sure who wished" → {{"is_negation": false, "converted_query": "not sure who wished"}}
- Query: "everyone except Amit" → {{"is_negation": true, "converted_query": "everyone"}}
"""
        
        try:
            # Call LLM based on provider
            if self.provider == "openai":
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content.strip()
            else:
                # Ollama
                response = self.ollama_client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    format="json"
                )
                result_text = response['message']['content'].strip()
            
            # Parse JSON response
            result = json.loads(result_text)
            is_negation = result.get('is_negation', False)
            converted_query = result.get('converted_query', query).strip()
            
            if is_negation:
                print(f"   🔄 Negation detected: '{query}' → '{converted_query}'")
            
            return is_negation, converted_query
        
        except Exception as e:
            print(f"   ⚠️ Negation detection failed: {e}")
            # Fallback: no negation, return original query
            return False, query
    
    def route(self, query: str) -> Dict[str, any]:
        """
        Main routing function using NER + LLM hybrid approach with Fuzzy Matching
        
        Logic:
        0. Apply fuzzy name matching (normalize names like 'amir' → 'aamir')
        1. Detect and convert negation (if present)
        2. IF NER finds entities AND LLM finds NO semantic → ANALYTICAL
        3. ELIF LLM finds semantic AND NER finds NO entities → SEMANTIC
        4. ELIF both find something → HYBRID
        5. ELSE neither found → UNCLEAR (ask for clarification)
        
        Args:
            query: Natural language query
        
        Returns:
            Routing decision with all extracted information
        """
        # Step 0: Apply fuzzy name matching
        query1 = query  # Original query
        query2 = query  # Will be normalized query
        fuzzy_applied = False
        fuzzy_replacements = []
        
        if self.smart_query_gen:
            query2, info = self.smart_query_gen.generate(query1)
            if info['normalized']:
                fuzzy_applied = True
                fuzzy_replacements = info['replacements']
                print(f"\n🔍 Fuzzy Matching Applied:")
                print(f"   query1 (original):   '{query1}'")
                print(f"   query2 (normalized): '{query2}'")
                for r in fuzzy_replacements:
                    print(f"   → '{r['original']}' matched to '{r['canonical']}'")
        
        # Use normalized query for routing
        working_query = query2
        
        # Step 1: Detect and convert negation
        is_negation, converted_query = self.detect_and_convert_negation(working_query)
        
        # Step 2: Extract entities with NER (use converted query)
        ner_result = self.extract_entities_with_ner(converted_query)
        
        # Step 3: Detect semantic concept with LLM (use converted query)
        llm_result = self.detect_semantic_with_llm(converted_query)
        
        # Step 4: Apply routing logic
        # Check entities from both NER and LLM
        has_entities_ner = ner_result['has_entities']
        has_entities_llm = bool(llm_result['person'])  # LLM detected person
        has_entities = has_entities_ner or has_entities_llm  # Either source
        has_semantic = llm_result['has_semantic']
        
        if has_entities and not has_semantic:
            query_type = 'ANALYTICAL'
            reasoning = "Entities found (persons/dates/numbers) but no semantic concept detected"
        
        elif has_semantic and not has_entities:
            query_type = 'SEMANTIC'
            reasoning = "Semantic concept detected but no entities (person/date) found"
        
        elif has_entities and has_semantic:
            query_type = 'HYBRID'
            reasoning = "Both entities (person/date) and semantic concept detected"
        
        else:
            # Neither found - ask for clarification
            query_type = 'UNCLEAR'
            reasoning = "Cannot determine query intent - please be more specific"
        
        # Step 4: Compile result
        result = {
            'query_type': query_type,
            'original_query': query,
            'normalized_query': query2,  # After fuzzy matching
            'converted_query': converted_query,  # For routing (positive form if negation)
            'is_negation': is_negation,  # Flag for set subtraction
            'reasoning': reasoning,
            
            # Fuzzy matching info
            'fuzzy_applied': fuzzy_applied,
            'fuzzy_replacements': fuzzy_replacements,
            
            # From NER
            'entities': ner_result['entities'],
            'year': ner_result['temporal_filters'].get('year'),
            'month': ner_result['temporal_filters'].get('month'),
            
            # From LLM
            'semantic_concept': llm_result['semantic_concept'],
            'person': llm_result['person'] or (ner_result['entities']['persons'][0] if ner_result['entities']['persons'] else None),
            'person_role': llm_result['person_role'],
            
            # Flags
            'has_entities': has_entities,
            'has_semantic': has_semantic
        }
        
        return result
    
    def format_result(self, result: Dict) -> str:
        """
        Format routing result as readable string
        
        Args:
            result: Router result
        
        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"Query Type: {result['query_type']}")
        lines.append(f"Reasoning: {result['reasoning']}")
        
        # Fuzzy matching info
        if result.get('fuzzy_applied'):
            lines.append(f"🔍 Fuzzy Match: '{result['original_query']}' → '{result['normalized_query']}'")
            for r in result.get('fuzzy_replacements', []):
                lines.append(f"   → '{r['original']}' matched to '{r['canonical']}'")
        
        if result.get('semantic_concept'):
            lines.append(f"💡 Semantic Concept: {result['semantic_concept']}")
        
        if result.get('person'):
            role_emoji = {'SENDER': '📤', 'RECIPIENT': '📥'}.get(result.get('person_role'), '👤')
            role = result.get('person_role') or 'UNKNOWN'
            lines.append(f"{role_emoji} Person: {result['person']} ({role})")
        
        if result.get('year') or result.get('month'):
            filters = []
            if result.get('year'):
                filters.append(f"year={result['year']}")
            if result.get('month'):
                filters.append(f"month={result['month']}")
            lines.append(f"📅 Filters: {', '.join(filters)}")
        
        if result.get('entities'):
            ent_summary = []
            for ent_type, values in result['entities'].items():
                if values:
                    ent_summary.append(f"{ent_type}={values}")
            if ent_summary:
                lines.append(f"🎯 Entities: {', '.join(ent_summary)}")
        
        return '\n'.join(lines)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NER + LLM Hybrid Router with Fuzzy Matching')
    parser.add_argument('folder_name', nargs='?', default='vellapanti', 
                        help='Folder name for name_pairs (e.g., parivar, vellapanti)')
    parser.add_argument('--provider', default='openai', choices=['openai', 'ollama'],
                        help='LLM provider')
    args = parser.parse_args()
    
    print("="*80)
    print("NER + LLM HYBRID ROUTER with FUZZY MATCHING - Testing")
    print("="*80)
    
    router = NERLLMHybridRouter(folder_name=args.folder_name, provider=args.provider)
    
    # Test queries covering all cases
    test_queries = [
        # ANALYTICAL: NER finds entities + LLM finds NO semantic
        ("who sent most messages in 2024?", "ANALYTICAL"),
        ("count messages per month", "ANALYTICAL"),
        ("top 5 senders in January", "ANALYTICAL"),
        ("how many messages did Amit send?", "ANALYTICAL"),
        
        # SEMANTIC: LLM finds semantic + NER finds NO entities
        ("show me funny messages", "SEMANTIC"),
        ("philosophical thoughts", "SEMANTIC"),
        ("birthday wishes", "SEMANTIC"),
        ("motivational quotes", "SEMANTIC"),
        
        # HYBRID: Both NER finds entities AND LLM finds semantic
        ("funny messages from Amit in January 2024", "HYBRID"),
        ("birthday wishes Priya received last month", "HYBRID"),
        ("philosophical messages by MKV in 2024", "HYBRID"),
        ("congratulations messages for Atul", "HYBRID"),
    ]
    
    results_table = []
    
    for query, expected_type in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Expected: {expected_type}")
        print('='*80)
        
        result = router.route(query)
        
        print(f"\n{router.format_result(result)}")
        
        # Check if matches expected
        match = "✅" if result['query_type'] == expected_type else "❌"
        print(f"\n{match} Classification: {result['query_type']} (expected: {expected_type})")
        
        results_table.append({
            'query': query,
            'expected': expected_type,
            'actual': result['query_type'],
            'match': match,
            'has_entities': result['has_entities'],
            'has_semantic': result['has_semantic']
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    correct = sum(1 for r in results_table if r['match'] == '✅')
    total = len(results_table)
    accuracy = (correct / total) * 100
    
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    print("\n" + "="*80)
    print("ROUTING LOGIC VERIFICATION")
    print("="*80)
    
    logic_table = """
    ┌────────────────────┬─────────────┬─────────────────┐
    │ Condition          │ Has Entities│ Has Semantic    │ Result      │
    ├────────────────────┼─────────────┼─────────────────┼─────────────┤
    │ Pure Analytical    │ YES ✅      │ NO ❌           │ ANALYTICAL  │
    │ Pure Semantic      │ NO ❌       │ YES ✅          │ SEMANTIC    │
    │ Hybrid             │ YES ✅      │ YES ✅          │ HYBRID      │
    │ Fallback           │ NO ❌       │ NO ❌           │ ANALYTICAL  │
    └────────────────────┴─────────────┴─────────────────┴─────────────┘
    
    Examples:
    ✅ "who sent most in 2024?" → entities(2024) + no-semantic → ANALYTICAL
    ✅ "funny messages" → no-entities + semantic(funny) → SEMANTIC
    ✅ "funny messages from Amit" → entities(Amit) + semantic(funny) → HYBRID
    """
    
    print(logic_table)
    
    print("\n" + "="*80)
    print("INTEGRATION GUIDE")
    print("="*80)
    print("""
    Use in whatsapp_embeddings_sentence_transformers.py:
    
    from sp6_ner_llm_hybrid_router_fuzzy_matching import NERLLMHybridRouter
    
    router = NERLLMHybridRouter(folder_name='vellapanti')  # or 'parivar'
    result = router.route(user_query)
    
    # Fuzzy matching is AUTOMATIC - check if applied:
    if result['fuzzy_applied']:
        print(f"Query normalized: {result['original_query']} -> {result['normalized_query']}")
    
    if result['query_type'] == 'SEMANTIC':
        # Pure vector search
        semantic_search(result['semantic_concept'])
    
    elif result['query_type'] == 'ANALYTICAL':
        # Pandas agent or analytical functions
        pandas_agent.query(result['normalized_query'])  # Use normalized!
    
    elif result['query_type'] == 'HYBRID':
        # Vector search + filters
        semantic_results = semantic_search(result['semantic_concept'])
        # Apply filters: result['person'], result['person_role'], result['year'], result['month']
        # Note: result['person'] is already the normalized name (e.g., 'aamir' not 'amir')
    """)

