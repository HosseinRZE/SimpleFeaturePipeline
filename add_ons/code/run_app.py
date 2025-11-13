#!/usr/bin/env python3
"""
Simple launcher script for the 2-Candle Volume Profile App
"""
import os
import sys
import subprocess

def main():
    print("🚀 Starting 2-Candle Volume Profile App")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this script from the app directory.")
        sys.exit(1)
    
    # Check if templates directory exists
    if not os.path.exists('templates/index.html'):
        print("❌ Error: templates/index.html not found.")
        sys.exit(1)
    
    print("✅ App structure verified")
    print("\n🌐 Starting Flask server...")
    print("   The app will be available at: http://localhost:5000")
    print("\n📋 To stop the server: Press Ctrl+C")
    print("=" * 50)
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thank you for using the Volume Profile App!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()