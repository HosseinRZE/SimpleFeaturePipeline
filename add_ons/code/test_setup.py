#!/usr/bin/env python3
"""
Simple test script to verify Flask app setup
"""
import sys
import os

def test_imports():
    """Test if required packages can be imported"""
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError:
        print("❌ Flask not installed. Run: pip install -r requirements.txt")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError:
        print("❌ Pandas not installed. Run: pip install -r requirements.txt")
        return False
    
    return True

def test_file_structure():
    """Test if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'templates/index.html'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def main():
    print("Testing 2-Candle Volume Profile App Setup")
    print("=" * 50)
    
    # Test imports
    print("\n1. Testing package imports...")
    imports_ok = test_imports()
    
    # Test file structure
    print("\n2. Testing file structure...")
    files_ok = test_file_structure()
    
    print("\n" + "=" * 50)
    if imports_ok and files_ok:
        print("✅ Setup verification complete! The app is ready to run.")
        print("\nTo start the application:")
        print("   python app.py")
        print("\nThen open http://localhost:5000 in your browser")
    else:
        print("❌ Setup verification failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()