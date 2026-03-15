#!/usr/bin/env python
"""
Verify WhatsApp QA Application Setup
Run this before starting the app to check all components
"""

import os
import sys
from pathlib import Path

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        return False
    return True

def check_packages():
    """Check required packages"""
    packages = {
        'streamlit': 'streamlit',
        'pandas': 'pd',
        'sentence_transformers': 'SentenceTransformer',
        'chromadb': 'chromadb',
        'spacy': 'spacy',
    }
    
    missing = []
    for package, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def check_data_files():
    """Check if data files exist"""
    folders = [
        "data/whatsapp/vellapanti/vellapanti_features.csv",
        "data/whatsapp/parivar/parivar_features.csv",
        "chroma_db_vellapanti",
        "chroma_db_parivar"
    ]
    
    found = {}
    for folder in folders:
        exists = Path(folder).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {folder}")
        found[folder] = exists
    
    missing = [f for f, exists in found.items() if not exists]
    if missing:
        print(f"\n⚠️  Missing files/folders:")
        for m in missing:
            print(f"  - {m}")
        print("\nCreate them with:")
        print("  python main_pipeline.py vellapanti vellapanti")
        print("  python sp5_whatsapp_embeddings_sentence_transformers.py vellapanti vellapanti")
        return len(missing) == 0 or any("parivar" in m for m in missing)  # At least one folder needed
    return True

def check_app_files():
    """Check if app files exist"""
    files = [
        "streamlit_qa_app.py",
        "sp6_ner_llm_hybrid_router_fuzzy_matching.py",
        "sp7_simple_pandas_agent.py",
        "sp8_optimized_hybrid_search.py",
        "requirements.txt"
    ]
    
    missing = []
    for file in files:
        exists = Path(file).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            missing.append(file)
    
    return len(missing) == 0

def check_imports():
    """Test if app imports work"""
    try:
        print("\n📋 Testing imports...")
        import streamlit
        print("✅ streamlit")
        import pandas
        print("✅ pandas")
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers")
        import chromadb
        print("✅ chromadb")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("🔍 WhatsApp QA Application - Verification Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python),
        ("Required Packages", check_packages),
        ("App Files", check_app_files),
        ("Data Files", check_data_files),
        ("Imports", check_imports),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n📌 {name}:")
        print("-" * 60)
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! Ready to run:")
        print("\n   streamlit run streamlit_qa_app.py")
    else:
        print("❌ Some checks failed. Fix issues above and try again.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
