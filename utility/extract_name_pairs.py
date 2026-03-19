"""
Extract Name Pairs for Query Normalization
1. Extract PERSON entities using NER
2. Find best matching pairs using SequenceMatcher
3. Output: {original_word: most_similar_word}
"""

import pandas as pd
import spacy
from difflib import SequenceMatcher
from collections import Counter
import json
import argparse
from pathlib import Path
from tqdm import tqdm


class NamePairExtractor:
    """
    Extracts person names and finds similar name pairs for typo correction
    """
    
    def __init__(self, csv_paths: list):
        """
        Args:
            csv_paths: List of CSV file paths to analyze
        """
        self.csv_paths = csv_paths
        
        print("=" * 80)
        print("NAME PAIR EXTRACTOR - NER + SequenceMatcher")
        print("=" * 80)
        
        # Load spaCy NER model
        print("\n[1/5] Loading spaCy NER model...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("✅ Loaded: en_core_web_sm")
        except OSError:
            print("⚠️  Model not found. Installing...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
            print("✅ Installed and loaded: en_core_web_sm")
        
        # Load CSV data
        print("\n[2/5] Loading CSV files...")
        self.dataframes = []
        for csv_path in csv_paths:
            if Path(csv_path).exists():
                df = pd.read_csv(csv_path)
                self.dataframes.append(df)
                print(f"✅ Loaded: {csv_path} ({len(df)} rows)")
            else:
                print(f"❌ Not found: {csv_path}")
        
        if not self.dataframes:
            raise FileNotFoundError("No CSV files found!")
        
        # Storage
        self.person_names = Counter()  # All person names with frequency
        self.name_pairs = {}  # Final output: {variant: canonical}
    
    def extract_person_entities(self):
        """
        Step 1: Extract PERSON entities using NER
        """
        print("\n[3/5] Extracting PERSON entities using NER...")
        
        for df in self.dataframes:
            # Source 1: Sender column (100% people)
            print("   → Extracting from 'sender' column...")
            for sender in df['sender'].unique():
                if sender and sender != 'system':
                    self._add_person_name(sender)
            
            # Source 2: Birthday person column (100% people)
            print("   → Extracting from 'birthday_person' column...")
            for person in df['birthday_person'].dropna().unique():
                self._add_person_name(person)
            
            # Source 3: NER from messages
            print("   → Extracting from messages using NER...")
            messages = df['message_english'].fillna('').tolist()
            
            # Process in batches for speed
            batch_size = 100
            total_batches = (len(messages) + batch_size - 1) // batch_size
            
            with tqdm(total=len(messages), desc="   Processing messages") as pbar:
                for doc in self.nlp.pipe(messages, batch_size=batch_size):
                    for ent in doc.ents:
                        if ent.label_ == 'PERSON':
                            self._add_person_name(ent.text)
                    pbar.update(1)
        
        # Print summary
        print(f"\n{'=' * 80}")
        print("EXTRACTED PERSON NAMES")
        print(f"{'=' * 80}")
        print(f"Total unique names: {len(self.person_names)}")
        print(f"\nTop 20 most mentioned:")
        for name, count in self.person_names.most_common(20):
            print(f"   {name:<20} : {count:>4} mentions")
        print(f"{'=' * 80}")
    
    def _add_person_name(self, name: str):
        """Clean and add person name to counter"""
        if not name:
            return
        
        # Split compound names: "Aamir Khan" → ["aamir", "khan"]
        words = name.strip().lower().split()
        
        for word in words:
            # Remove non-alphabetic characters (emojis, numbers)
            clean_word = ''.join(c for c in word if c.isalpha())
            
            # Skip very short or very long names
            if 2 < len(clean_word) < 20:
                self.person_names[clean_word] += 1
    
    def find_best_pairs(self, similarity_threshold: float = 0.75):
        """
        Step 2: Find best matching pairs using SequenceMatcher
        Each word maps to ONLY its closest match
        """
        print(f"\n[4/5] Finding best matching pairs (threshold={similarity_threshold})...")
        
        names = list(self.person_names.keys())
        n = len(names)
        
        print(f"   Comparing {n} names ({n * (n-1) // 2} comparisons)...")
        
        # Store all similarities for each name
        name_similarities = {name: [] for name in names}
        
        # Calculate similarities between all pairs
        comparisons_done = 0
        total_comparisons = n * (n - 1) // 2
        
        with tqdm(total=total_comparisons, desc="   Comparing names") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    name1 = names[i]
                    name2 = names[j]
                    
                    # Calculate string similarity
                    similarity = SequenceMatcher(None, name1, name2).ratio()
                    
                    # Only consider if above threshold
                    if similarity >= similarity_threshold:
                        name_similarities[name1].append((name2, similarity))
                        name_similarities[name2].append((name1, similarity))
                    
                    pbar.update(1)
        
        # For each name, find the BEST match
        print("\n   Selecting best match for each name...")
        
        potential_pairs = []
        
        for name, matches in name_similarities.items():
            if matches:
                # Sort by similarity (highest first)
                matches.sort(key=lambda x: x[1], reverse=True)
                best_match, best_similarity = matches[0]
                
                # Determine canonical form (more frequent = canonical)
                freq_name = self.person_names[name]
                freq_match = self.person_names[best_match]
                
                if freq_name < freq_match:
                    # name is variant, best_match is canonical
                    variant = name
                    canonical = best_match
                else:
                    # best_match is variant, name is canonical
                    variant = best_match
                    canonical = name
                
                potential_pairs.append({
                    'variant': variant,
                    'canonical': canonical,
                    'similarity': best_similarity,
                    'freq_variant': min(freq_name, freq_match),
                    'freq_canonical': max(freq_name, freq_match)
                })
        
        # Remove duplicates (keep highest similarity for each variant)
        seen_variants = {}
        for pair in sorted(potential_pairs, key=lambda x: x['similarity'], reverse=True):
            variant = pair['variant']
            if variant not in seen_variants:
                seen_variants[variant] = pair
        
        # Create final pairs dictionary
        self.name_pairs = {}
        for variant, pair in seen_variants.items():
            # Don't map a word to itself
            if pair['variant'] != pair['canonical']:
                self.name_pairs[pair['variant']] = pair['canonical']
        
        # Print results
        print(f"\n{'=' * 80}")
        print("SIMILAR NAME PAIRS FOUND")
        print(f"{'=' * 80}")
        print(f"{'Variant':<20} → {'Canonical':<20} {'Similarity':<12} {'Freq (v→c)'}")
        print(f"{'-' * 80}")
        
        # Sort by similarity for display
        sorted_pairs = sorted(
            [(v, c, seen_variants[v]['similarity'], 
              seen_variants[v]['freq_variant'], 
              seen_variants[v]['freq_canonical']) 
             for v, c in self.name_pairs.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        for variant, canonical, similarity, freq_v, freq_c in sorted_pairs:
            print(f"{variant:<20} → {canonical:<20} {similarity:.2%}        {freq_v:>3} → {freq_c:<3}")
        
        print(f"{'-' * 80}")
        print(f"Total pairs found: {len(self.name_pairs)}")
        print(f"{'=' * 80}")
        
        return self.name_pairs
    
    def save_pairs(self, output_path: str = 'name_pairs.json'):
        """
        Step 3: Save pairs to JSON file
        """
        print(f"\n[5/5] Saving to {output_path}...")
        
        output = {
            "description": "Name variation pairs for query normalization",
            "usage": "Replace 'variant' with 'canonical' in user queries",
            "threshold_used": 0.75,
            "pairs": self.name_pairs,
            "all_names": list(self.person_names.keys()),
            "statistics": {
                "total_unique_names": len(self.person_names),
                "total_pairs": len(self.name_pairs)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved to: {output_path}")
        print(f"\n{'=' * 80}")
        print("USAGE EXAMPLE")
        print(f"{'=' * 80}")
        print("""
# Load pairs
with open('name_pairs.json', 'r') as f:
    data = json.load(f)
    name_pairs = data['pairs']

# Normalize query
def normalize_query(query, pairs):
    words = query.lower().split()
    normalized = [pairs.get(word, word) for word in words]
    return ' '.join(normalized)

# Example:
query = "who talked about amir khan"
normalized = normalize_query(query, name_pairs)
# Result: "who talked about aamir khan"
""")
        print(f"{'=' * 80}\n")
        
        return output
    
    def interactive_review(self):
        """
        Let user review and approve/edit pairs
        """
        if not self.name_pairs:
            print("No pairs to review. Run find_best_pairs() first.")
            return
        
        print(f"\n{'=' * 80}")
        print("INTERACTIVE REVIEW")
        print(f"{'=' * 80}")
        print("Review each pair. Enter:")
        print("  'y' = approve")
        print("  'n' = reject")
        print("  'e' = edit canonical form")
        print("  'q' = quit review")
        print(f"{'=' * 80}\n")
        
        approved_pairs = {}
        
        for variant, canonical in list(self.name_pairs.items()):
            print(f"\n'{variant}' → '{canonical}'")
            choice = input("  Approve? (y/n/e/q): ").strip().lower()
            
            if choice == 'q':
                print("\n⏹️  Review stopped. Keeping approved pairs.")
                break
            elif choice == 'y':
                approved_pairs[variant] = canonical
                print("  ✅ Approved")
            elif choice == 'e':
                new_canonical = input(f"  Enter correct form for '{variant}': ").strip().lower()
                if new_canonical and new_canonical != variant:
                    approved_pairs[variant] = new_canonical
                    print(f"  ✅ Updated: '{variant}' → '{new_canonical}'")
                else:
                    print("  ❌ Rejected (invalid input)")
            else:
                print("  ❌ Rejected")
        
        self.name_pairs = approved_pairs
        print(f"\n✅ Review complete. {len(approved_pairs)} pairs approved.")


def main():
    parser = argparse.ArgumentParser(description='Extract Name Pairs using NER + SequenceMatcher')
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
        '--threshold',
        type=float,
        default=0.75,
        help='Similarity threshold (0.0-1.0, default: 0.75)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='name_pairs.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Review pairs interactively'
    )
    
    args = parser.parse_args()
    
    # Construct CSV path and output path
    csv_path = f'data/whatsapp/{args.folder_name}/{args.file_name}_features.csv'
    output_path = args.output if args.output != 'name_pairs.json' else f'name_pairs_{args.folder_name}.json'
    
    print(f"\nInput CSV: {csv_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_path}\n")
    
    # Run extraction
    extractor = NamePairExtractor([csv_path])
    extractor.extract_person_entities()
    extractor.find_best_pairs(similarity_threshold=args.threshold)
    
    if args.interactive:
        extractor.interactive_review()
    
    extractor.save_pairs(output_path)
    
    print("✅ Done!")


if __name__ == "__main__":
    main()
