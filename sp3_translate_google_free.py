# Input: csv file parivar_split.csv 
# text to english translation using deep-translator library (Google Translate)
# with custom glossary support. input glossary from input_data/input_data.txt 
# also protects terms in [DO_NOT_TRANSLATE] section from being translated
# Output: parivar_google_translated.csv


import pandas as pd
from deep_translator import GoogleTranslator
import sys
import time
import re
sys.stdout.reconfigure(encoding="utf-8")

# Initialize translator (auto-detect source language, translate to English)
translator = GoogleTranslator(source='auto', target='en')

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
        
        # STEP 2: Send to Google Translate (deep-translator)
        translation = translator.translate(text_with_placeholders)
        
        # STEP 3: Replace placeholders back with English terms
        final_translation = restore_placeholders(translation, replacements)
        
        return final_translation
    except Exception as e:
        print(f"Error translating '{text[:30]}...': {e}")
        return text

# Process INPUT CSV
def f_google_translate(folder_name, file_name):
    csv_path = f"data/whatsapp/{folder_name}/{file_name}_split.csv"
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
    output_path = f"data/whatsapp/{folder_name}/{file_name}_google_translated.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ Translation complete!")
    print(f"✓ Saved to: {output_path}\n")

