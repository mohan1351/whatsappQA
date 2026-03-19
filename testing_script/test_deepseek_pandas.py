"""
Test DeepSeek-Coder 6.7B with Ollama for Pandas Code Generation
100% FREE, runs locally with GPU acceleration
"""

import ollama


def test_deepseek_pandas(query):
    """Test DeepSeek with a pandas query"""
    
    prompt = f"""Generate ONLY pandas code (no explanation) to answer this question.

DataFrame 'df' has these columns:
- timestamp: Message timestamp (datetime)
- sender: Name of the person who sent the message
- message: Original message text
- message_english: English translation of the message
- dt: Parsed datetime object
- year: Year when message was sent (int)
- month: Month number (1-12)
- month_short: Short month name (Jan, Feb, etc.)
- month_year: Combined month and year
- day_month: Day of the month
- birthday_person: Name of person whose birthday falls on that date (if any)

Question: {query}

Pandas code (starting with df):"""
    
    try:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        print("🤖 DeepSeek-Coder generating code...\n")
        
        response = ollama.generate(
            model='deepseek-coder:6.7b',
            prompt=prompt,
            options={'temperature': 0.2}
        )
        
        code = response['response'].strip()
        print(f"✅ Generated Code:\n{code}\n")
        
        return code
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Same queries as langchain_pandas_agent.py for comparison
    test_queries = [
        "who was quiet in 2025",
        "who did not send birthday messages to MKV in 2024?",
        "who all sent birthday wished to MKV in 2024?",
    ]
    
    print("="*80)
    print("TESTING DEEPSEEK-CODER 6.7B (Ollama)")
    print("="*80)
    print("\nModel: deepseek-coder:6.7b")
    print("Cost: $0.00 (100% FREE)")
    print("Privacy: 100% local, no data sent anywhere")
    print("Speed: GPU accelerated (RTX 3050)")
    print("="*80)
    
    for query in test_queries:
        test_deepseek_pandas(query)
