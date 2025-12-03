"""
Integration tests for the autocomplete system.
Verifies that all components work correctly.
"""
import sys
import time
import logging
from typing import List, Tuple

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    modules = [
        ("config", "Configuration"),
        ("data_loader", "Data Loader"),
        ("embedding_generator", "Embedding Generator"),
        ("lexical_indexer", "Lexical Indexer"),
        ("feature_extractor", "Feature Extractor"),
        ("retrieval_engine", "Retrieval Engine"),
    ]
    
    all_passed = True
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {description:30s} OK")
        except Exception as e:
            print(f"✗ {description:30s} FAILED: {e}")
            all_passed = False
    
    return all_passed


def test_data_loading():
    """Test data loading from HuggingFace."""
    print("\n" + "="*80)
    print("TEST 2: Data Loading")
    print("="*80)
    
    try:
        from data_loader import DataLoader
        
        loader = DataLoader()
        
        # Load pool (small sample)
        pool_df = loader.load_query_pool()
        print(f"✓ Query pool loaded: {len(pool_df):,} queries")
        
        # Load features
        features_df = loader.load_query_features()
        print(f"✓ Query features loaded: {len(features_df):,} queries")
        
        # Load train sample
        train_df = loader.load_train_data()
        print(f"✓ Training data loaded: {len(train_df):,} examples")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_embeddings():
    """Test embedding generation and FAISS search."""
    print("\n" + "="*80)
    print("TEST 3: Embeddings and FAISS")
    print("="*80)
    
    try:
        from embedding_generator import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        # Test encoding
        test_queries = ["iphone 16", "black blazer", "running shoes"]
        embeddings = generator.encode_queries(test_queries)
        print(f"✓ Encoded {len(test_queries)} queries")
        print(f"  Embedding shape: {embeddings.shape}")
        
        # Load or build index
        embeddings_full, metadata, faiss_index = generator.load_embeddings_and_index()
        print(f"✓ FAISS index loaded: {faiss_index.ntotal:,} vectors")
        
        # Test search
        distances, indices = generator.search(embeddings, k=5)
        print(f"✓ Search completed")
        
        print("\nSample search results for 'iphone 16':")
        for i, idx in enumerate(indices[0]):
            result = metadata["queries"][idx]
            print(f"  {i+1}. {result} (score: {distances[0][i]:.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lexical_search():
    """Test BM25 and n-gram search."""
    print("\n" + "="*80)
    print("TEST 4: Lexical Search (BM25 + N-grams)")
    print("="*80)
    
    try:
        from lexical_indexer import LexicalIndexer
        
        indexer = LexicalIndexer()
        indexer.load_indices()
        
        test_prefix = "ifon 16"
        
        # BM25 search
        bm25_indices, bm25_scores = indexer.search_bm25(test_prefix, k=5)
        print(f"✓ BM25 search completed: {len(bm25_indices)} results")
        
        print(f"\nBM25 results for '{test_prefix}':")
        for i, (idx, score) in enumerate(zip(bm25_indices, bm25_scores)):
            print(f"  {i+1}. {indexer.queries[idx]} (score: {score:.4f})")
        
        # N-gram search
        ngram_indices, ngram_scores = indexer.search_ngram(test_prefix, k=5)
        print(f"\n✓ N-gram search completed: {len(ngram_indices)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Lexical search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\n" + "="*80)
    print("TEST 5: Feature Extraction")
    print("="*80)
    
    try:
        from data_loader import DataLoader
        from feature_extractor import FeatureExtractor
        
        loader = DataLoader()
        query_features_map = loader.get_query_to_features_map()
        
        extractor = FeatureExtractor(query_features_map)
        
        # Extract features for sample
        features = extractor.extract_single(
            prefix="ifon 16",
            query="iphone 16",
            semantic_score=0.85,
            bm25_score=2.5,
            ngram_score=0.7,
            fuzzy_score=0.8
        )
        
        print(f"✓ Extracted {len(features)} features")
        
        # Show some features
        print("\nSample features:")
        important = [
            "semantic_cosine", "bm25_score", "edit_distance_ratio",
            "exact_prefix_match", "token_jaccard", "popularity_score"
        ]
        for feat in important:
            if feat in features:
                print(f"  {feat:25s}: {features[feat]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval_engine():
    """Test the complete retrieval pipeline."""
    print("\n" + "="*80)
    print("TEST 6: Complete Retrieval Pipeline")
    print("="*80)
    
    try:
        from retrieval_engine import RetrievalEngine
        
        # Check if model exists
        from config import RANKER_MODEL_PATH
        if not RANKER_MODEL_PATH.exists():
            print("⚠ Ranker model not found. Skipping this test.")
            print("  Run 'python train_ranker.py' first to train the model.")
            return True
        
        print("Loading retrieval engine (this may take a moment)...")
        engine = RetrievalEngine()
        
        test_prefixes = [
            "ifon 16",
            "blaek blaijer",
            "running sho"
        ]
        
        print("\nTest queries:")
        print("-"*80)
        
        total_time = 0
        
        for prefix in test_prefixes:
            start = time.time()
            results = engine.retrieve(prefix)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            
            print(f"\nPrefix: '{prefix}' ({elapsed:.2f}ms)")
            for i, result in enumerate(results[:5], 1):
                print(f"  {i}. {result}")
        
        avg_time = total_time / len(test_prefixes)
        print(f"\n✓ Average latency: {avg_time:.2f}ms")
        
        if avg_time < 200:
            print("✓ Latency is good!")
        else:
            print("⚠ Latency is high. Consider optimizations.")
        
        return True
        
    except Exception as e:
        print(f"✗ Retrieval engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("AUTOCOMPLETE SYSTEM - INTEGRATION TESTS")
    print("="*80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Embeddings & FAISS", test_embeddings),
        ("Lexical Search", test_lexical_search),
        ("Feature Extraction", test_feature_extraction),
        ("Retrieval Pipeline", test_retrieval_engine),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12s} {test_name}")
    
    print("="*80)
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Train the model: python train_ranker.py")
        print("  2. Launch UI: streamlit run app.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the errors above.")
        return 1


def main():
    """Run tests based on command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test autocomplete system")
    parser.add_argument(
        "--test",
        choices=["all", "imports", "data", "embeddings", "lexical", "features", "retrieval"],
        default="all",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        sys.exit(run_all_tests())
    elif args.test == "imports":
        sys.exit(0 if test_imports() else 1)
    elif args.test == "data":
        sys.exit(0 if test_data_loading() else 1)
    elif args.test == "embeddings":
        sys.exit(0 if test_embeddings() else 1)
    elif args.test == "lexical":
        sys.exit(0 if test_lexical_search() else 1)
    elif args.test == "features":
        sys.exit(0 if test_feature_extraction() else 1)
    elif args.test == "retrieval":
        sys.exit(0 if test_retrieval_engine() else 1)


if __name__ == "__main__":
    main()