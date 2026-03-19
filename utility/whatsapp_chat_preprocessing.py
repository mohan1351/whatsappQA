"""
WhatsApp Chat Preprocessing for Embeddings
Cleans and normalizes messages WITHOUT modifying original CSV data
Use this before creating embeddings to improve retrieval quality
"""

import re
import emoji


def whatsapp_chat_preprocessing(text):
    """
    Preprocess WhatsApp message for embedding creation
    Does NOT modify original CSV - only returns cleaned text for embeddings
    
    Args:
        text (str): Original WhatsApp message from 'message_english' column
    
    Returns:
        str or None: Preprocessed text ready for embedding, or None if message should be filtered
    
    Preprocessing Steps:
        1. Remove URLs (noise)
        2. Remove email addresses
        3. Remove phone numbers
        4. Normalize emojis (keep max 3 unique)
        5. Normalize excessive punctuation (!!! → !)
        6. Remove duplicate words
        7. Remove extra whitespace
        8. Filter very short messages (< 3 chars or < 2 words)
    
    Example:
        >>> original = "Check this!!! 🎂🎉🎂🎉 http://example.com"
        >>> cleaned = whatsapp_chat_preprocessing(original)
        >>> print(cleaned)
        "Check this! 🎂🎉"
    """
    
    # Step 0: Validate input
    if not text or not isinstance(text, str) or text.strip() == '':
        return None
    
    # Step 1: Remove URLs (noise for embeddings)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Step 2: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Step 3: Remove phone numbers (Indian and international formats)
    text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)
    
    # Step 4: Normalize emojis (keep max 3 unique)
    try:
        seen_emojis = []
        clean_chars = []
        
        for char in text:
            if char in emoji.EMOJI_DATA:
                # Keep only first occurrence of each emoji (max 3 total)
                if char not in seen_emojis and len(seen_emojis) < 3:
                    seen_emojis.append(char)
            else:
                clean_chars.append(char)
        
        # Rebuild: text + unique emojis at end
        text = ''.join(clean_chars).strip()
        if seen_emojis:
            text += ' ' + ''.join(seen_emojis)
    except:
        # If emoji processing fails, continue without emoji normalization
        pass
    
    # Step 5: Normalize excessive punctuation (!!! → !, ??? → ?)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Step 6: Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Step 7: Remove duplicate consecutive words
    words = text.split()
    deduplicated = []
    prev_word = None
    
    for word in words:
        word_lower = word.lower()
        if word_lower != prev_word:
            deduplicated.append(word)
            prev_word = word_lower
    
    text = ' '.join(deduplicated)
    
    # Step 8: Final cleanup
    text = text.strip()
    
    # Step 9: Filter out messages that are too short or meaningless
    if len(text) < 3:
        return None
    
    if len(text.split()) < 2:
        return None
    
    # Step 10: Filter junk patterns (case-insensitive)
    junk_patterns = [
        'media omitted',
        'this message was deleted',
        'image omitted',
        'video omitted',
        'audio omitted',
        'sticker omitted',
        'gif omitted',
        'document omitted'
    ]
    
    text_lower = text.lower()
    for pattern in junk_patterns:
        if pattern in text_lower:
            return None
    
    return text


def batch_preprocess(messages):
    """
    Preprocess multiple messages at once
    
    Args:
        messages (list): List of WhatsApp messages
    
    Returns:
        list: List of preprocessed messages (None values filtered out)
    
    Example:
        >>> messages = ["Hello!!! 🎂🎂", "Check http://example.com", "Ok"]
        >>> cleaned = batch_preprocess(messages)
        >>> print(cleaned)
        ["Hello! 🎂"]
    """
    preprocessed = []
    for msg in messages:
        cleaned = whatsapp_chat_preprocessing(msg)
        if cleaned:  # Only keep non-None results
            preprocessed.append(cleaned)
    
    return preprocessed


def preprocess_dataframe(df, message_column='message_english'):
    """
    Preprocess WhatsApp messages in a pandas DataFrame
    Creates a NEW column 'message_preprocessed' without modifying original
    
    Args:
        df (pd.DataFrame): DataFrame with WhatsApp messages
        message_column (str): Name of column containing messages (default: 'message_english')
    
    Returns:
        pd.DataFrame: DataFrame with new 'message_preprocessed' column
    
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("whatsapp.csv")
        >>> df = preprocess_dataframe(df)
        >>> # Now df has both 'message_english' (original) and 'message_preprocessed' (cleaned)
    """
    import pandas as pd
    
    # Apply preprocessing to create NEW column
    df['message_preprocessed'] = df[message_column].apply(whatsapp_chat_preprocessing)
    
    # Filter out None values (rows where preprocessing returned None)
    df_cleaned = df[df['message_preprocessed'].notna()].copy()
    
    print(f"Original messages: {len(df)}")
    print(f"After preprocessing: {len(df_cleaned)} ({len(df) - len(df_cleaned)} filtered out)")
    
    return df_cleaned


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("WHATSAPP CHAT PREPROCESSING - TEST EXAMPLES")
    print("="*80)
    
    test_cases = [
        "Happy birthday 🎂🎉🎂🎉🎂🎉!!!!",
        "Check this out: https://www.example.com/product?id=123",
        "Call me at +91-98765-43210",
        "Email: user@example.com",
        "AMAZING AMAZING AMAZING!!!!!!",
        "Thank you thank you so much much much",
        "Ok",
        "👍",
        "<Media omitted>",
        "This message was deleted",
        "I love this product!!! 😊😊😊😊😊",
        "Hello    world\n\n\nHow   are    you?"
    ]
    
    print("\n" + "-"*80)
    print("Test Cases:")
    print("-"*80)
    
    for i, original in enumerate(test_cases, 1):
        cleaned = whatsapp_chat_preprocessing(original)
        print(f"\n{i}. Original: {repr(original)}")
        print(f"   Cleaned:  {repr(cleaned)}")
        if cleaned is None:
            print(f"   Status:   ❌ FILTERED OUT")
        else:
            print(f"   Status:   ✅ KEPT")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print("\nUsage in embedding script:")
    print("""
    from whatsapp_chat_preprocessing import whatsapp_chat_preprocessing
    
    # Load messages
    messages = df['message_english'].tolist()
    
    # Preprocess for embeddings (original CSV unchanged!)
    messages_clean = [whatsapp_chat_preprocessing(msg) for msg in messages]
    messages_clean = [msg for msg in messages_clean if msg]  # Remove None
    
    # Create embeddings with clean messages
    embeddings = model.encode(messages_clean)
    """)