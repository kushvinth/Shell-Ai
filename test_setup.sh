#!/bin/bash

echo "🧪 Testing the CI/CD pipeline setup locally..."

# Check if required files exist
echo "📁 Checking required files..."
if [ ! -f "notebook.ipynb" ]; then
    echo "❌ notebook.ipynb not found!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

if [ ! -f "dataset/train.csv" ]; then
    echo "❌ dataset/train.csv not found!"
    exit 1
fi

if [ ! -f "dataset/test.csv" ]; then
    echo "❌ dataset/test.csv not found!"
    exit 1
fi

echo "✅ All required files found"

# Test Python dependencies
echo "🐍 Testing Python dependencies..."
python -c "
import sys
required_packages = [
    'pandas', 'numpy', 'scikit-learn', 'lightgbm', 
    'xgboost', 'scipy', 'papermill', 'jupyter'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'❌ {package}')

if missing_packages:
    print(f'\\n📦 Missing packages: {missing_packages}')
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('\\n✅ All required packages are available')
"

echo "🎯 Setup verification complete!"
echo "💡 The CI/CD pipeline should now work correctly."
echo ""
echo "🚀 To trigger the pipeline:"
echo "   1. Push changes to main branch"
echo "   2. Manually trigger via GitHub Actions UI"
echo "   3. Wait for scheduled run (daily at 7 AM UTC)"
