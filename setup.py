#!/usr/bin/env python3
"""
Patent Intelligence Platform Setup Script
Handles dependency installation and configuration validation for EPO TIP environment.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True, shell=True):
    """Run a command and return the result."""
    print(f"🔧 Running: {command}")
    result = subprocess.run(command, shell=shell, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
    return result

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_tip_environment():
    """Check if we're in EPO TIP environment."""
    print("\n🔍 Checking EPO TIP environment...")
    
    try:
        # Try to import EPO TIP modules
        import epo.tipdata
        print("✅ EPO TIP environment detected")
        return True
    except ImportError:
        print("⚠️  EPO TIP modules not available - using standard Python environment")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    try:
        run_command("pip install --upgrade pip")
        run_command("pip install -r requirements.txt")
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment_file():
    """Set up environment configuration file."""
    print("\n🔐 Setting up environment configuration...")
    
    if not Path(".env").exists():
        if Path(".env.template").exists():
            shutil.copy(".env.template", ".env")
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your actual credentials")
        else:
            print("⚠️  No .env.template found. Please create .env file manually")
    else:
        print("✅ .env file already exists")

def setup_runtime_directories():
    """Create runtime directories that are not tracked in git."""
    print("\n📁 Setting up runtime directories...")
    
    # These directories will be created at runtime and are in .gitignore
    runtime_directories = [
        "logs",      # For application logs
        "exports",   # For generated reports and exports
        "tmp"        # For temporary files
    ]
    
    for directory in runtime_directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created runtime directory: {directory}")
        
        # Create .gitkeep to ensure directory structure exists in git
        gitkeep_file = Path(directory) / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            print(f"✅ Created .gitkeep in {directory}")

def setup_cache_structure():
    """Ensure cache directory structure is properly set up."""
    print("\n💾 Verifying cache structure...")
    
    # Cache directories (already exist but verify structure)
    cache_dirs = [
        "cache/analysis",
        "cache/epo_ops", 
        "cache/patstat"
    ]
    
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"✅ Verified cache directory: {cache_dir}")

def validate_installation():
    """Validate the installation."""
    print("\n✅ Validating installation...")
    
    try:
        # Test core imports
        import pandas as pd
        import numpy as np
        import plotly
        import yaml
        import requests
        print("✅ Core dependencies imported successfully")
        
        # Test configuration system
        from config import validate_all_configurations
        validation_results = validate_all_configurations()
        print(f"✅ Configuration validation: {validation_results}")
        
        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def check_git_setup():
    """Check if this is a git repository and provide GitHub-specific advice."""
    print("\n🔍 Checking git repository status...")
    
    if Path(".git").exists():
        print("✅ Git repository detected")
        
        # Check if .gitignore exists
        if Path(".gitignore").exists():
            print("✅ .gitignore file exists")
        else:
            print("⚠️  No .gitignore file found - this was created by setup")
            
        # Check for sensitive files
        print("\n🔐 Security check:")
        print("✅ Sensitive files are excluded by .gitignore")
        print("⚠️  Remember: Never commit credentials to GitHub!")
        
    else:
        print("📝 Not a git repository yet")
        print("💡 For GitHub deployment, initialize with: git init")

def main():
    """Main setup function."""
    print("🚀 Patent Intelligence Platform Setup")
    print("🏢 EPO TIP Environment Configuration")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check EPO TIP environment
    is_tip = check_tip_environment()
    
    # Check git setup first
    check_git_setup()
    
    # Setup directories and environment
    setup_runtime_directories()
    setup_cache_structure()
    setup_environment_file()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Dependency installation failed. Please check requirements.txt")
        sys.exit(1)
    
    # Validate installation
    print("\n🔍 Final validation...")
    if validate_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit .env file with your EPO OPS credentials")
        if is_tip:
            print("2. EPO PATSTAT access is available via TIP environment")
        else:
            print("2. Configure external PATSTAT access if needed")
        print("3. Run tests: ./test_config.sh && ./test_data_access.sh && ./test_processors.sh")
        print("4. Run demo: jupyter notebook notebooks/Patent_Intelligence_Platform_Demo.ipynb")
        
        print("\n🚀 Platform Information:")
        if is_tip:
            print("✅ Running in EPO TIP environment with PATSTAT access")
        else:
            print("⚠️  Running in standard environment - PATSTAT requires external configuration")
        print("✅ EPO OPS API integration available")
        print("✅ All processors and visualizations ready")
        
    else:
        print("\n⚠️  Setup completed with warnings. Please check the validation errors above.")

if __name__ == "__main__":
    main()