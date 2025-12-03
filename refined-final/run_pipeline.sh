#!/bin/bash

# Smart Autocomplete System - Setup Script
# This script sets up the entire system from scratch

set -e  # Exit on error

echo "=========================================="
echo "Smart Autocomplete System Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if GPU is available
print_status "Checking for GPU..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    print_status "GPU detected!"
    GPU_AVAILABLE=true
else
    print_warning "No GPU detected. Will use CPU (slower)."
    GPU_AVAILABLE=false
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_warning "Virtual environment already exists."
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
print_status "Installing dependencies..."
echo "This may take a few minutes..."

if [ "$GPU_AVAILABLE" = true ]; then
    print_status "Installing with GPU support..."
    pip install -r requirements.txt
else
    print_warning "Installing with CPU support..."
    # Replace faiss-gpu with faiss-cpu
    sed 's/faiss-gpu/faiss-cpu/' requirements.txt | pip install -r /dev/stdin
fi

print_status "Dependencies installed successfully!"

echo ""
echo "=========================================="
echo "Building Indices"
echo "=========================================="
echo ""

# Download and cache data
print_status "Downloading dataset from HuggingFace..."
python3 data_loader.py

# Generate embeddings
print_status "Generating embeddings (this may take 5-10 minutes)..."
python3 embedding_generator.py

# Build lexical indices
print_status "Building BM25 and n-gram indices..."
python3 lexical_indexer.py

print_status "Indices built successfully!"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Train the ranker:"
echo "   ${GREEN}python train_ranker.py${NC}"
echo ""
echo "2. Generate predictions:"
echo "   ${GREEN}python generate_submission.py --mode submission${NC}"
echo ""
echo "3. Launch Streamlit UI:"
echo "   ${GREEN}streamlit run app.py${NC}"
echo ""
echo "4. For SSH users, create tunnel from local machine:"
echo "   ${GREEN}ssh -L 8501:localhost:8501 your-server${NC}"
echo "   Then open: ${GREEN}http://localhost:8501${NC}"
echo ""
echo "=========================================="
echo "For help, see README.md"
echo "=========================================="