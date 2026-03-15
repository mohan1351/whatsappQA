"""
Smart Query Generator
Takes query1 → extracts PERSON entities via NER → normalizes using name_pairs.json → returns query2
"""

import json
import spacy
from pathlib import Path
from difflib import SequenceMatcher


class SmartQueryGenerator:
    """
    Generates normalized query by:
    1. Extracting PERSON entities from query using spaCy NER
    2. Finding closest match in all_names from name_pairs_{folder_name}.json
    3. Replacing with the best matching name
    """
    
    def __init__(self, folder_name: str, threshold: float = 0.75):
        """
        Args:
            folder_name: Folder name (e.g., 'parivar', 'vellapanti')
            threshold: String similarity threshold for matching (0-1)
        """
        self.folder_name = folder_name
        self.threshold = threshold
        self.json_path = f"name_pairs_{folder_name}.json"
        self.all_names = self._load_all_names(self.json_path)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _load_all_names(self, json_path: str) -> list:
        """Load all_names list from JSON for fuzzy matching"""
        path = Path(json_path)
        if not path.exists():
            print(f"⚠️  {json_path} not found. Normalization disabled.")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_names = data.get('all_names', [])
        print(f"✅ Loaded {len(all_names)} names for fuzzy matching")
        return all_names
    
    def _extract_persons(self, query: str) -> list[tuple[str, int, int]]:
        """
        Extract PERSON entities from query using NER
        
        Returns:
            List of (entity_text, start_idx, end_idx)
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(query)
        persons = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.append((ent.text, ent.start_char, ent.end_char))
        
        return persons
    
    def _find_canonical(self, name: str) -> str | None:
        """
        Find best matching name from all_names using string similarity
        Handles multi-word names by matching each word separately.
        
        Args:
            name: Name from query (could be "amir khan" with typo)
            
        Returns:
            Best matching name(s) combined (e.g., "aamir khan"), or None
        """
        if not self.all_names:
            return None
        
        # Split into words and match each
        words = name.split()
        normalized_words = []
        any_change = False
        
        for word in words:
            word_lower = word.lower()
            best_match = word  # Default to original
            best_score = 0
            
            for known_name in self.all_names:
                score = SequenceMatcher(None, word_lower, known_name.lower()).ratio()
                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_match = known_name
            
            # Check if we found a better match (different from original)
            if best_match.lower() != word_lower and best_score >= self.threshold:
                any_change = True
            
            normalized_words.append(best_match)
        
        if not any_change:
            return None
        
        return ' '.join(normalized_words)
    
    def generate(self, query1: str) -> tuple[str, dict]:
        """
        Generate normalized query2 from query1
        
        Args:
            query1: Original query
            
        Returns:
            Tuple of (query2, info)
            - query2: Normalized query
            - info: Dictionary with details
        """
        query1 = query1.strip()
        
        # Extract PERSON entities
        persons = self._extract_persons(query1)
        
        if not persons:
            return query1, {
                'normalized': False,
                'reason': 'no PERSON entities found',
                'entities': [],
                'replacements': []
            }
        
        # Build query2 by replacing persons with canonical names
        query2 = query1
        replacements = []
        offset = 0  # Track position shift due to replacements
        
        for person_text, start, end in persons:
            canonical = self._find_canonical(person_text)
            
            if canonical and canonical.lower() != person_text.lower():
                # Replace in query2
                adjusted_start = start + offset
                adjusted_end = end + offset
                query2 = query2[:adjusted_start] + canonical + query2[adjusted_end:]
                
                # Update offset for next replacement
                offset += len(canonical) - len(person_text)
                
                replacements.append({
                    'original': person_text,
                    'canonical': canonical
                })
        
        if not replacements:
            return query1, {
                'normalized': False,
                'reason': 'no matching variants in name_pairs',
                'entities': [p[0] for p in persons],
                'replacements': []
            }
        
        return query2, {
            'normalized': True,
            'reason': 'PERSON entities normalized',
            'entities': [p[0] for p in persons],
            'replacements': replacements
        }


def get_normalized_query(query1: str, folder_name: str) -> str:
    """
    Simple function to get normalized query
    
    Usage:
        from smart_query_generator import get_normalized_query
        query2 = get_normalized_query("who is amir khan", "parivar")
        
    Args:
        query1: Original query
        folder_name: Folder name (e.g., 'parivar', 'vellapanti')
        
    Returns:
        query2 (normalized query)
    """
    generator = SmartQueryGenerator(folder_name=folder_name)
    query2, _ = generator.generate(query1)
    return query2


def get_query_pair(query1: str, folder_name: str) -> tuple[str, str]:
    """
    Get both original and normalized query
    
    Usage:
        from smart_query_generator import get_query_pair
        query1, query2 = get_query_pair("who is amir khan", "parivar")
        
    Returns:
        Tuple of (query1, query2)
    """
    generator = SmartQueryGenerator(folder_name=folder_name)
    query2, _ = generator.generate(query1)
    return query1, query2


# Demo when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Query Generator')
    parser.add_argument('folder_name', type=str, help='Folder name (e.g., parivar, vellapanti)')
    parser.add_argument('query', nargs='?', help='Query to normalize')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    generator = SmartQueryGenerator(folder_name=args.folder_name)
    
    if args.interactive:
        print("\n" + "="*60)
        print("INTERACTIVE MODE - Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                query1 = input("Enter query1: ").strip()
                if query1.lower() in ['quit', 'exit', 'q']:
                    break
                if not query1:
                    continue
                
                query2, info = generator.generate(query1)
                
                print(f"\n  query1: {query1}")
                print(f"  query2: {query2}")
                print(f"  Entities found: {info['entities']}")
                if info['replacements']:
                    for r in info['replacements']:
                        print(f"    '{r['original']}' → '{r['canonical']}'")
                print()
                
            except KeyboardInterrupt:
                break
    
    elif args.query:
        query2, info = generator.generate(args.query)
        print(f"\nquery1: {args.query}")
        print(f"query2: {query2}")
        print(f"\nEntities: {info['entities']}")
        print(f"Normalized: {info['normalized']}")
        if info['replacements']:
            print("Replacements:")
            for r in info['replacements']:
                print(f"  '{r['original']}' → '{r['canonical']}'")
    
    else:
        # Demo
        samples = [
            "who is amir khan",
            "what did amit say about birthday",
            "messages from deepak"
        ]
        print("\n" + "="*60)
        print("DEMO")
        print("="*60)
        for q in samples:
            query2, info = generator.generate(q)
            status = "✅" if info['normalized'] else "⚪"
            print(f"\n{status} query1: {q}")
            print(f"   query2: {query2}")
