# run_streamlit_app.py
# Simple launcher for the Streamlit Research Assistant

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'ollama',
        'ddgs',
        'trafilatura',
        'sentence_transformers',
        'scikit-learn',
        'langgraph'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        import ollama
        # Try to list models
        models = ollama.list()
        if not models.get('models'):
            print("No Ollama models found!")
            print("Download a model first: ollama pull llama3")
            return False
        
        print("Ollama is running with models:")
        for model in models['models']:
            print(f"   - {model['name']}")
        return True
        
    except Exception as e:
        print("Ollama connection failed!")
        print("Start Ollama first: ollama serve")
        return False

def main():
    """Launch the Streamlit application"""
    print("Starting AI Research Assistant...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\nTip: Make sure Ollama is running and you have downloaded a model")
        print("   Start Ollama: ollama serve")
        print("   Download model: ollama pull llama3")
        sys.exit(1)
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "research-assistant.py"
    
    if not streamlit_app.exists():
        print(f"Cannot find research-assistant.py in {script_dir}")
        sys.exit(1)
    
    print("All dependencies ready!")
    print("Launching Streamlit app...")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app),
            "--theme.primaryColor=#667eea",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f0f2f6",
            "--theme.textColor=#262730",
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    except subprocess.CalledProcessError:
        print("Failed to launch Streamlit app")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n App stopped by user")

if __name__ == "__main__":
    main()