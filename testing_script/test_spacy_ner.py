"""Quick test to verify spaCy NER installation"""
import spacy

print("Testing spaCy installation...")

try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except OSError:
    print("⚠️  Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model installed")

# Test entity extraction
test_query = "funny messages from Amit in January 2024"
doc = nlp(test_query)

print(f"\nTest query: {test_query}")
print("Entities found:")
for ent in doc.ents:
    print(f"  - {ent.text} ({ent.label_})")

print("\n✅ spaCy NER test complete!")
