#!/usr/bin/env python3
"""
Setup script for Marketing Budget Optimization Chatbot
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openai>=1.12.0",
        "pydantic>=2.0.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install '{dep}'", f"Installing {dep}"):
            return False
    
    return True

def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OpenAI API key not found!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print("✅ OpenAI API key is set")
    return True

def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    try:
        result = subprocess.run([sys.executable, "test_chatbot.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ All tests passed!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Some tests failed:")
        print(e.stdout)
        if e.stderr:
            print("Errors:")
            print(e.stderr)
        return False

def main():
    """Main setup function"""
    print("🚀 Marketing Budget Optimization Chatbot Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    print("\n📦 Installing Dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check OpenAI API key
    print("\n🔑 Checking OpenAI API Key...")
    api_key_set = check_openai_key()
    
    # Run tests
    print("\n🧪 Running Tests...")
    tests_passed = run_tests()
    
    # Final status
    print("\n" + "=" * 60)
    print("📊 Setup Summary")
    print("=" * 60)
    print(f"✅ Dependencies installed")
    print(f"{'✅' if api_key_set else '⚠️ '} OpenAI API key {'set' if api_key_set else 'NOT SET'}")
    print(f"{'✅' if tests_passed else '❌'} Tests {'passed' if tests_passed else 'failed'}")
    
    if api_key_set and tests_passed:
        print("\n🎉 Setup complete! You can now run the chatbot:")
        print("python marketing_chatbot.py")
    else:
        print("\n⚠️  Setup incomplete. Please address the issues above.")
        if not api_key_set:
            print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        if not tests_passed:
            print("2. Check the test output above for specific issues")

if __name__ == "__main__":
    main()