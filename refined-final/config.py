"""
Configuration file for autocomplete system.
All paths, hyperparameters, and settings in one place.
"""
from pathlib import Path

# ============================================================================
# PATHS - All artifacts saved here for quick reloading
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

# Create directories
for dir_path in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Artifact paths
POOL_EMBEDDINGS_PATH = ARTIFACTS_DIR / "pool_embeddings.npy"
POOL_METADATA_PATH = ARTIFACTS_DIR / "pool_metadata.pkl"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"
NGRAM_INDEX_PATH = ARTIFACTS_DIR / "ngram_index.pkl"
TRAIN_FEATURES_PATH = ARTIFACTS_DIR / "train_features.pkl"
RANKER_MODEL_PATH = MODELS_DIR / "lightgbm_ranker.txt"

# Cache for inference
INFERENCE_CACHE_PATH = ARTIFACTS_DIR / "inference_cache.pkl"

# ============================================================================
# MODEL SETTINGS
# ============================================================================
# Sentence transformer model - fast and good quality
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Output dimension for all-MiniLM-L6-v2

# GPU settings
USE_GPU = True  # Set to True to use GPU for embeddings
BATCH_SIZE = 256  # Batch size for embedding generation (increase if you have more GPU memory)

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================
# Number of candidates to retrieve from each method
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
NGRAM_TOP_K = 10
FUZZY_TOP_K = 10

# Combined candidate pool size before reranking
MAX_CANDIDATES = 50

# Final output size
FINAL_TOP_K = 10

# FAISS settings
FAISS_HNSW_M = 32  # Number of neighbors in HNSW graph
FAISS_EF_CONSTRUCTION = 200  # Construction time parameter
FAISS_EF_SEARCH = 64  # Search time parameter

# BM25 parameters
BM25_K1 = 1.2
BM25_B = 0.75

# Fuzzy matching
FUZZY_THRESHOLD = 0.7  # Minimum similarity for fuzzy matches
MIN_PREFIX_LENGTH_FOR_FUZZY = 3

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
# Feature normalization
NORMALIZE_FEATURES = True

# Popularity features (will be log-transformed)
POPULARITY_FEATURES = ["orders", "volume", "catalog_clicks", "catalog_views"]

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
# Negative sampling
NUM_HARD_NEGATIVES = 5  # From semantic retrieval
NUM_BM25_NEGATIVES = 3  # From BM25
NUM_RANDOM_NEGATIVES = 2  # Random from pool
TOTAL_NEGATIVES_PER_POSITIVE = 10

# LightGBM Ranker hyperparameters
RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "importance_type": "gain",
}

# Early stopping
EARLY_STOPPING_ROUNDS = 50
VALIDATION_SPLIT = 0.1  # Hold out 10% for validation

# Relevance labels (graded)
LABEL_EXACT_MATCH = 3
LABEL_HIGH_ENGAGEMENT = 2  # Has orders
LABEL_MEDIUM_ENGAGEMENT = 1  # Has clicks/views
LABEL_NEGATIVE = 0

# ============================================================================
# PREPROCESSING
# ============================================================================
# Text normalization
LOWERCASE = True
REMOVE_SPECIAL_CHARS = True
TOKENIZE_METHOD = "whitespace"  # or "wordpiece"

# Minimum lengths
MIN_PREFIX_LENGTH = 2
MIN_QUERY_LENGTH = 2

# ============================================================================
# INFERENCE OPTIMIZATIONS
# ============================================================================
# Cache settings
ENABLE_CACHE = True
CACHE_SIZE = 10000  # Number of prefixes to cache
CACHE_POPULAR_PREFIXES = True  # Pre-cache popular prefixes

# Parallel processing
PARALLEL_RETRIEVAL = True  # Run semantic and BM25 in parallel

# Short prefix handling
SHORT_PREFIX_THRESHOLD = 2  # Use popularity-only for prefixes <= this length
SHORT_PREFIX_TOP_N = 1000  # Number of popular queries to consider for short prefixes

# ============================================================================
# HUGGINGFACE DATASET
# ============================================================================
HF_DATASET_NAME = "123tushar/Dice_Challenge_2025"
HF_TRAIN_FILES = "train_data/*.parquet"
HF_TEST_FILES = "test_prefixes_data/*.parquet"
HF_FEATURES_FILES = "query_features/*.parquet"
HF_POOL_FILES = "pool/*.parquet"

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "autocomplete.log"

# ============================================================================
# STREAMLIT SETTINGS
# ============================================================================
STREAMLIT_PORT = 8501
STREAMLIT_TITLE = "Smart Autocomplete System"
STREAMLIT_SHOW_SCORES = True  # Show detailed scores in UI
STREAMLIT_SHOW_FEATURES = False  # Show features (debug mode)