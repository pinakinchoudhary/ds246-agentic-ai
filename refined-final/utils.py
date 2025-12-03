"""
Utility functions for system management and diagnostics.
"""
import os
import sys
from pathlib import Path
import json
import logging
from typing import Dict, List
import psutil

from config import *

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check if GPU is available and print details."""
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU Available: {gpu_count} device(s)")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
            
            # Check current usage
            if gpu_count > 0:
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"  Current usage:")
                print(f"    Allocated: {memory_allocated:.2f} GB")
                print(f"    Reserved: {memory_reserved:.2f} GB")
            
            return True
        else:
            print("✗ No GPU available. Will use CPU.")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed. Cannot check GPU.")
        return False


def check_artifacts():
    """Check which artifacts are already built."""
    artifacts = {
        "Pool Embeddings": POOL_EMBEDDINGS_PATH,
        "Pool Metadata": POOL_METADATA_PATH,
        "FAISS Index": FAISS_INDEX_PATH,
        "BM25 Index": BM25_INDEX_PATH,
        "N-gram Index": NGRAM_INDEX_PATH,
        "Training Features": TRAIN_FEATURES_PATH,
        "Ranker Model": RANKER_MODEL_PATH,
    }
    
    print("\nArtifact Status:")
    print("=" * 60)
    
    total_size = 0
    
    for name, path in artifacts.items():
        if path.exists():
            size = path.stat().st_size / (1024 ** 2)  # MB
            total_size += size
            print(f"✓ {name:25s} {size:8.2f} MB")
        else:
            print(f"✗ {name:25s} Not found")
    
    print("=" * 60)
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    return artifacts


def check_system_resources():
    """Check system resources (CPU, RAM, disk)."""
    print("\nSystem Resources:")
    print("=" * 60)
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu_count} cores, {cpu_percent:.1f}% used")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.total / 1e9:.2f} GB total")
    print(f"     {ram.available / 1e9:.2f} GB available ({ram.percent:.1f}% used)")
    
    # Disk
    disk = psutil.disk_usage(str(BASE_DIR))
    print(f"Disk: {disk.total / 1e9:.2f} GB total")
    print(f"      {disk.free / 1e9:.2f} GB free ({disk.percent:.1f}% used)")
    
    print("=" * 60)


def estimate_memory_requirements():
    """Estimate memory requirements for the system."""
    print("\nEstimated Memory Requirements:")
    print("=" * 60)
    
    # Assuming typical sizes
    pool_size = 5_000_000  # 5M queries
    embedding_dim = EMBEDDING_DIM
    
    embeddings_size = pool_size * embedding_dim * 4 / 1e9  # float32
    faiss_size = embeddings_size * 1.2  # Slightly larger with HNSW
    bm25_size = pool_size * 100 / 1e9  # Rough estimate
    
    print(f"Embeddings: ~{embeddings_size:.2f} GB")
    print(f"FAISS Index: ~{faiss_size:.2f} GB")
    print(f"BM25 Index: ~{bm25_size:.2f} GB")
    print(f"Model Runtime: ~1-2 GB")
    print("=" * 60)
    print(f"Total Estimate: ~{embeddings_size + faiss_size + bm25_size + 2:.2f} GB")
    print("=" * 60)


def clean_artifacts(confirm: bool = False):
    """Remove all generated artifacts."""
    if not confirm:
        response = input("Are you sure you want to delete all artifacts? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    print("\nCleaning artifacts...")
    
    paths_to_clean = [
        ARTIFACTS_DIR,
        MODELS_DIR,
        DATA_DIR / "cache",
    ]
    
    import shutil
    
    for path in paths_to_clean:
        if path.exists():
            shutil.rmtree(path)
            print(f"✓ Removed {path}")
            path.mkdir(exist_ok=True, parents=True)
    
    print("\nAll artifacts cleaned. Run setup to rebuild.")


def verify_installation():
    """Verify that all required packages are installed."""
    required_packages = [
        "torch",
        "numpy",
        "pandas",
        "sentence_transformers",
        "faiss",
        "lightgbm",
        "streamlit",
        "datasets",
        "rank_bm25",
        "rapidfuzz",
    ]
    
    print("\nVerifying Installation:")
    print("=" * 60)
    
    all_installed = True
    
    for package in required_packages:
        try:
            if package == "faiss":
                # Try both faiss-gpu and faiss-cpu
                try:
                    import faiss
                    version = faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
                    print(f"✓ {package:25s} {version}")
                except:
                    print(f"✗ {package:25s} Not installed")
                    all_installed = False
            else:
                module = __import__(package)
                version = module.__version__ if hasattr(module, '__version__') else "unknown"
                print(f"✓ {package:25s} {version}")
        except ImportError:
            print(f"✗ {package:25s} Not installed")
            all_installed = False
    
    print("=" * 60)
    
    if all_installed:
        print("✓ All packages installed successfully!")
    else:
        print("✗ Some packages are missing. Run: pip install -r requirements.txt")
    
    return all_installed


def run_diagnostics():
    """Run complete system diagnostics."""
    print("=" * 80)
    print("AUTOCOMPLETE SYSTEM DIAGNOSTICS")
    print("=" * 80)
    
    # Check installation
    verify_installation()
    
    # Check GPU
    print("\n")
    check_gpu_availability()
    
    # Check resources
    check_system_resources()
    
    # Check artifacts
    check_artifacts()
    
    # Estimate requirements
    estimate_memory_requirements()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


def main():
    """CLI for utility functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autocomplete System Utilities")
    parser.add_argument(
        "command",
        choices=["check", "clean", "verify", "diagnostics"],
        help="Command to run"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    if args.command == "check":
        check_artifacts()
    elif args.command == "clean":
        clean_artifacts(confirm=args.confirm)
    elif args.command == "verify":
        verify_installation()
    elif args.command == "diagnostics":
        run_diagnostics()


if __name__ == "__main__":
    main()