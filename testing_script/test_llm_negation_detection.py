"""
Test LLM-based Negation Detection
Demonstrates LLM accurately detects true negation vs false positives
"""

from ner_llm_hybrid_router import NERLLMHybridRouter
from dotenv import load_dotenv

load_dotenv()


def test_llm_negation():
    """
    Test LLM negation detection with various query types
    """
    print("="*80)
    print("LLM-BASED NEGATION DETECTION TEST")
    print("="*80)
    
    # Initialize router
    router = NERLLMHybridRouter()
    
    # Test cases
    test_cases = [
        {
            'query': 'who did not wish Krati birthday in 2025',
            'expected': True,
            'reason': 'True negation: "did NOT" means negation'
        },
        {
            'query': 'who didn\'t send birthday wishes',
            'expected': True,
            'reason': 'True negation: "didn\'t" is contraction of "did not"'
        },
        {
            'query': 'messages not from Amit',
            'expected': True,
            'reason': 'True negation: excluding Amit\'s messages'
        },
        {
            'query': 'everyone except Priya',
            'expected': True,
            'reason': 'True negation: "except" means exclusion'
        },
        {
            'query': 'who never replied to messages',
            'expected': True,
            'reason': 'True negation: "never" means absence of action'
        },
        {
            'query': 'who did wish Krati birthday in 2025',
            'expected': False,
            'reason': 'NOT negation: "did" is auxiliary verb (positive query)'
        },
        {
            'query': 'did anyone wish Krati happy birthday',
            'expected': False,
            'reason': 'NOT negation: "did" is auxiliary verb in question (positive)'
        },
        {
            'query': 'not sure which messages to show',
            'expected': False,
            'reason': 'FALSE POSITIVE: user uncertainty, not query negation'
        },
        {
            'query': 'messages about not giving up',
            'expected': False,
            'reason': 'FALSE POSITIVE: "not" is part of topic/content'
        },
        {
            'query': 'I\'m not looking for birthday wishes',
            'expected': False,
            'reason': 'FALSE POSITIVE: user clarifying intent, not data negation'
        },
        {
            'query': 'show me messages, not statistics',
            'expected': False,
            'reason': 'FALSE POSITIVE: clarifying format preference'
        },
        {
            'query': 'funny messages from Amit',
            'expected': False,
            'reason': 'Normal query: no negation keywords at all'
        },
        {
            'query': 'who sent birthday wishes to Krati',
            'expected': False,
            'reason': 'Normal query: positive form'
        }
    ]
    
    results = {
        'passed': 0,
        'failed': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {test['query']}")
        print(f"Expected: is_negation={test['expected']} ({test['reason']})")
        
        # Detect negation
        is_negation, converted_query = router.detect_and_convert_negation(test['query'])
        
        print(f"Got: is_negation={is_negation}")
        if is_negation:
            print(f"Converted: \"{converted_query}\"")
        
        # Verify result
        if is_negation == test['expected']:
            print("✅ PASS")
            results['passed'] += 1
        else:
            print("❌ FAIL")
            results['failed'] += 1
            
            if is_negation and not test['expected']:
                print("   → False Positive: Detected negation when there wasn't one")
                results['false_positives'] += 1
            elif not is_negation and test['expected']:
                print("   → False Negative: Missed actual negation")
                results['false_negatives'] += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"   False Positives: {results['false_positives']}")
    print(f"   False Negatives: {results['false_negatives']}")
    print(f"\nAccuracy: {results['passed']/len(test_cases)*100:.1f}%")
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ LLM correctly distinguishes true negation from false positives")
    else:
        print(f"\n⚠️ {results['failed']} test(s) failed")


if __name__ == "__main__":
    try:
        test_llm_negation()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
