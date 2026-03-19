"""
Test Semantic Keyword Expansion
Shows how we automatically expand any keyword to include variations
"""

import ollama


def expand_semantic_concept(keyword: str) -> str:
    """
    Automatically expand a keyword to include semantic variations
    Uses Ollama DeepSeek-Coder to generate common variations
    """
    try:
        prompt = f"""List semantic variations and common abbreviations for: "{keyword}"

Requirements:
- Include common abbreviations (e.g., HBD for birthday, LOL for funny)
- Include casual variations (e.g., "b'day", "bday")
- Include formal variations (e.g., "warm wishes", "felicitations")
- Keep it concise (5-10 variations max)
- Output ONLY the variations separated by spaces

Example:
Input: birthday
Output: birthday wishes happy birthday HBD b'day bday celebration happy bday

Now generate for: {keyword}
Output:"""
        
        response = ollama.generate(
            model='deepseek-coder:6.7b',
            prompt=prompt,
            options={'temperature': 0.3}
        )
        
        expanded = response['response'].strip()
        
        # Clean up if needed
        if len(expanded.split()) > 15 or '\n' in expanded:
            return f"{keyword} wishes messages {keyword}"
        
        return expanded
        
    except Exception as e:
        print(f"Error: {e}")
        return f"{keyword} wishes messages {keyword}"


def test_expansion(keyword: str):
    """Test expansion for a specific keyword"""
    print(f"\n{'='*80}")
    print(f"Original: {keyword}")
    print('='*80)
    
    expanded = expand_semantic_concept(keyword)
    
    print(f"Expanded: {expanded}")
    print(f"\nVariations found: {expanded.split()}")
    print(f"Count: {len(expanded.split())} terms")


if __name__ == "__main__":
    print("="*80)
    print("SEMANTIC KEYWORD EXPANSION - General Solution")
    print("="*80)
    print("\nThis automatically expands ANY keyword to include variations")
    print("Uses Ollama DeepSeek-Coder for intelligent expansion")
    
    # Test various keywords
    test_keywords = [
        "birthday",
        "anniversary",
        "philosophical",
        "funny",
        "congratulations",
        "condolence",
        "good morning",
        "thank you"
    ]
    
    for keyword in test_keywords:
        test_expansion(keyword)
    
    print("\n" + "="*80)
    print("BENEFITS")
    print("="*80)
    print("""
✅ No hardcoding - works for ANY concept
✅ Captures abbreviations (HBD, LOL, GN, TY, etc.)
✅ Includes casual variations (b'day, bday, hbd, etc.)
✅ Includes formal variations (felicitations, warm wishes, etc.)
✅ Automatically adapts to context
✅ FREE - uses local Ollama model
    """)
    
    print("\n" + "="*80)
    print("EXAMPLE WORKFLOW")
    print("="*80)
    print("""
Query: "who wished happy anniversary in 2024?"

Step 1: Extract semantic concept
   → "anniversary"

Step 2: Expand automatically using Ollama
   → "anniversary wishes happy anniversary celebration milestone happy anni"

Step 3: Semantic search with expanded query
   → Finds messages with:
      - "anniversary"
      - "happy anniversary" 
      - "anni" (casual)
      - "celebration" (related concept)
      - "milestone" (related concept)

Step 4: Filter by year 2024 and group by sender
   → Result: Who wished anniversary in 2024

✅ Works for birthdays, anniversaries, condolences, congratulations, etc.!
    """)
