"""
Simple WhatsApp QA Pipeline
Processes any WhatsApp chat from .txt to features CSV
"""

from sp1_whatsapp_data_processing import f_txt_to_csv
from sp2_datadelimited import clean_to_split
from sp3_translate_google_free import f_google_translate
from sp4_feature_building import add_features_to_whatsapp_data


def run_pipeline(folder_name, file_name):
    """
    Run complete WhatsApp data processing pipeline
    
    Args:
        folder_name: Folder name in data/whatsapp/ (e.g., 'parivar', 'vellapanti')
        file_name: File name without extension (e.g., 'parivar', 'chat')
    
    Example:
        run_pipeline('parivar', 'parivar')
        # Processes: data/whatsapp/parivar/parivar.txt
    """
    
    print("="*80)
    print(f"WHATSAPP PIPELINE - Processing: {folder_name}/{file_name}")
    print("="*80)
    
    try:
        # Step 1: Parse raw chat.txt to clean CSV
        print("\n[STEP 1/4] Parsing WhatsApp chat...")
        f_txt_to_csv(folder_name, file_name)
        
        # Step 2: Split into columns (timestamp, sender, message)
        print("\n[STEP 2/4] Splitting data fields...")
        clean_to_split(folder_name, file_name)
        
        # Step 3: Translate to English
        print("\n[STEP 3/4] Translating to English...")
        print("⏳ This may take several minutes...")
        f_google_translate(folder_name, file_name)
        
        # Step 4: Add features (year, month, birthday_person)
        print("\n[STEP 4/4] Building features...")
        add_features_to_whatsapp_data(folder_name, file_name)
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE!")
        print(f"📁 Output: data/whatsapp/{folder_name}/{file_name}_features.csv")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) >= 3:
        folder = sys.argv[1]
        file = sys.argv[2]
    else:
        # Default: vellapanti
        folder = "vellapanti"
        file = "vellapanti"
        print(f"💡 Usage: python main_pipeline.py <folder_name> <file_name>")
        print(f"   Using default: {folder}/{file}")
    
    # Run pipeline
    success = run_pipeline(folder, file)
    
    if success:
        print(f"\n🚀 Next step: Create embeddings")
        print(f"   python sp5_whatsapp_embeddings_sentence_transformers.py")
        print(f"\n💬 Then run Q&A interface:")
        print(f"   python sp8_optimized_hybrid_search.py")
