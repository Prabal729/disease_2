#!/usr/bin/env python3
"""
Deployment helper script for Render.com
This script prepares the application for deployment by:
1. Checking file structure
2. Validating imports
3. Creating deployment package
"""

import os
import sys
from pathlib import Path

def check_deployment_files():
    """Check if all required deployment files exist"""
    required_files = [
        'app.py',
        'requirements-deploy.txt',
        'runtime.txt',
        'render.yaml',
        'src/shared.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required deployment files found")
        return True

def check_pages_structure():
    """Check if pages directory exists and has required files"""
    pages_dir = Path('pages')
    if not pages_dir.exists():
        print("❌ Pages directory not found")
        return False
    
    page_files = list(pages_dir.glob('*.py'))
    if len(page_files) < 4:
        print(f"❌ Expected at least 4 page files, found {len(page_files)}")
        return False
    
    print(f"✅ Found {len(page_files)} page files")
    return True

def validate_imports():
    """Test if imports work correctly"""
    try:
        # Test main app import
        sys.path.insert(0, str(Path('src')))
        from shared import set_page, inject_theme, load_artifacts
        print("✅ Shared module imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main deployment check"""
    print("🚀 Running deployment checks...")
    print("=" * 50)
    
    checks = [
        check_deployment_files(),
        check_pages_structure(),
        validate_imports()
    ]
    
    if all(checks):
        print("=" * 50)
        print("✅ All checks passed! Ready for deployment.")
        print("\n📋 Deployment checklist:")
        print("1. Push code to GitHub repository")
        print("2. Connect repository to Render.com")
        print("3. Use 'requirements-deploy.txt' as requirements file")
        print("4. Set start command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true")
        print("5. Deploy!")
    else:
        print("=" * 50)
        print("❌ Some checks failed. Please fix issues before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()

