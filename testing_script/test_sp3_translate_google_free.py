# Input: csv file parivar_split.csv 
# text to english translation using googletrans library
# with custom glossary support. input glossary from input_data/input_data.txt 
# also protects terms in [DO_NOT_TRANSLATE] section from being translated
# Output: parivar_google_translated.csv


import pandas as pd
from googletrans import Translator
import sys
import time
import re
sys.stdout.reconfigure(encoding="utf-8")

translator = Translator()

def load_glossary(file_path="input_data/input_data.txt"):
    """Load custom glossary from text file"""
    glossary = {}
    do_not_translate = {}
    in_do_not_translate_section = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Check for DO_NOT_TRANSLATE section
                if line == '[DO_NOT_TRANSLATE]':
                    in_do_not_translate_section = True
                    continue
                
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    if in_do_not_translate_section:
                        # These terms should NOT be translated - keep as-is
                        term = line.strip()
                        pattern = re.escape(term).replace(r'\ ', r'\s+')
                        pattern = r'\b' + pattern + r'\b'
                        do_not_translate[term] = {
                            'pattern': pattern,
                            'preserve': term  # Keep original term
                        }
                    elif ',' in line:
                        # These are Hindi -> English translation mappings
                        hindi, english = line.split(',', 1)
                        hindi_term = hindi.strip()
                        english_term = english.strip()
                        pattern = re.escape(hindi_term).replace(r'\ ', r'\s+')
                        pattern = r'\b' + pattern + r'\b'
                        glossary[hindi_term] = {
                            'pattern': pattern,
                            'english': english_term
                        }
        
        print(f"✓ Loaded {len(glossary)} translation terms from {file_path}")
        print(f"✓ Loaded {len(do_not_translate)} protected terms (will not be translated)\n")
    except FileNotFoundError:
        print(f"Warning: Glossary file '{file_path}' not found. Using without custom glossary.\n")
    
    return glossary, do_not_translate

# Load custom glossary from file
HINDI_GLOSSARY, DO_NOT_TRANSLATE = load_glossary()

def replace_with_placeholders(text):
    """Replace glossary terms with unique placeholders BEFORE translation"""
    replacements = {}  # Track {placeholder: replacement_term}
    modified_text = text
    idx = 0
    
    # First, protect DO_NOT_TRANSLATE terms (keep them exactly as-is)
    for term, term_info in DO_NOT_TRANSLATE.items():
        # Use numbers only - Google won't translate pure numbers
        placeholder = f"99{idx:03d}99"
        if re.search(term_info['pattern'], modified_text, flags=re.IGNORECASE):
            # Preserve the original term exactly
            modified_text = re.sub(term_info['pattern'], placeholder, modified_text, flags=re.IGNORECASE)
            replacements[placeholder] = term_info['preserve']
            idx += 1
    
    # Then, handle translation terms (Hindi -> English)
    for hindi_term, term_info in HINDI_GLOSSARY.items():
        # Use numbers only - Google won't translate pure numbers
        placeholder = f"88{idx:03d}88"
        if re.search(term_info['pattern'], modified_text, flags=re.IGNORECASE):
            modified_text = re.sub(term_info['pattern'], placeholder, modified_text, flags=re.IGNORECASE)
            replacements[placeholder] = term_info['english']
            idx += 1
    
    return modified_text, replacements

def restore_placeholders(text, replacements):
    """Replace placeholders with English terms AFTER translation"""
    for placeholder, english_term in replacements.items():
        # Direct replacement - numeric placeholders won't be modified by Google
        text = text.replace(placeholder, english_term)
    return text

def translate_with_google(text):
    """Translate Romanized Hindi to English using Google Translate with protected glossary terms"""
    if pd.isna(text) or str(text).strip() == "" or str(text).startswith("<Media"):
        return text
    
    try:
        # STEP 1: Replace glossary terms with placeholders (Google won't translate these)
        text_with_placeholders, replacements = replace_with_placeholders(str(text))
        
        # STEP 2: Send to Google Translate
        result = translator.translate(text_with_placeholders, src='hi', dest='en')
        translation = result.text
        
        # STEP 3: Replace placeholders back with English terms
        final_translation = restore_placeholders(translation, replacements)
        
        return final_translation
    except Exception as e:
        print(f"Error translating '{text[:30]}...': {e}")
        return text

# Test examples
print("--- Testing Google Translation with Custom Glossary ---")
test_examples = [
    "bandar kya jaane angur ka swad",
    "main kal market jaunga",
    "Ms Bilkul yeh bahut accha hai",
    "kya haal hai bhai",
    "Ms Bilkul added you",
    "Ms Bilkul added preeti mummy jio"
    "mama ghar aa rahe hain",  # Should translate as "maternal uncle"
    "mami ne khana banaya"     # Should translate as "maternal aunt"
]

for example in test_examples:
    translation = translate_with_google(example)
    print(f"IN:  {example}")
    print(f"OUT: {translation}\n")
    time.sleep(0.5)  # Small delay to avoid rate limiting

# Process INPUT CSV
csv_path = "data/whatsapp/parivar/parivar_split.csv"
print(f"Loading CSV: {csv_path}")
df = pd.read_csv(csv_path, encoding="utf-8-sig")

print(f"Found {len(df)} messages to translate\n")

# Translate all messages with progress
translations = []
for idx, message in enumerate(df["message"]):
    if idx % 5 == 0:
        print(f"Progress: {idx}/{len(df)}")
    
    translated = translate_with_google(message)
    translations.append(translated)
    time.sleep(0.3)  # Delay to avoid rate limits

df["message_english"] = translations

# Save results
output_path = "data/whatsapp/parivar/parivar_google_translated.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✓ Translation complete!")
print(f"✓ Saved to: {output_path}\n")

# Show sample results
print("--- Sample Translations ---")
sample_count = 0
for idx, row in df.iterrows():
    if row["message"] != row["message_english"] and sample_count < 10:
        print(f"\n[{row['sender']}]")
        print(f"Original:   {row['message']}")
        print(f"Translated: {row['message_english']}")
        sample_count += 1
