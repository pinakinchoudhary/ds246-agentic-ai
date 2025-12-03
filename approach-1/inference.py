"""
Standalone Inference Script for Query Auto-completion
Usage: python inference.py --query "your search prefix"
"""

import pickle
import numpy as np
import pandas as pd
import faiss
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz
from sentence_transformers import SentenceTransformer
from numba import jit
import argparse

# Load all model components
MODEL_DIR = Path('model_artifacts')

print("Loading model components...")

# Load BM25
with open(MODEL_DIR / 'bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# Load PCA
pca = joblib.load(MODEL_DIR / 'pca_model.pkl')

# Load config
with open(MODEL_DIR / 'model_config.pkl', 'rb') as f:
    config = pickle.load(f)

# Load Sentence Transformer
model = SentenceTransformer(config['sentence_transformer_model'])

# Load FAISS index
faiss_index = faiss.read_index(str(MODEL_DIR / 'faiss_index.bin'))

# Load pool data
df_pool_enhanced = pd.read_parquet(MODEL_DIR / 'pool_data.parquet')
pool_dense_embeddings = np.load(MODEL_DIR / 'pool_dense_embeddings.npy')
pool_sparse_matrix = load_npz(MODEL_DIR / 'pool_sparse_matrix.npz')

# Load pool queries
with open(MODEL_DIR / 'pool_queries.pkl', 'rb') as f:
    pool_queries = pickle.load(f)

print(f"✅ Model loaded successfully!")
print(f"   - Pool size: {len(pool_queries)} queries")
print(f"   - FAISS index: {faiss_index.ntotal} vectors")
print(f"   - Embedding dim: {config['target_dim']}D")


@jit(nopython=True)
def combine_scores_fast(dense_scores, sparse_scores, popularity_scores, popularity_weight):
    """Fast score combination using Numba JIT"""
    return dense_scores + sparse_scores + popularity_scores * popularity_weight


def hybrid_score_norm(dense, sparse, alpha=0.5):
    """Normalize dense and sparse vectors for hybrid search"""
    # Normalize dense vector
    dense_norm = np.array(dense) / (np.linalg.norm(dense) + 1e-10)
    
    # Normalize sparse vector
    sparse_values = np.array(sparse['values'])
    sparse_norm = sparse_values / (np.linalg.norm(sparse_values) + 1e-10)
    
    # Apply alpha weighting
    hdense = (dense_norm * alpha).astype(np.float32)
    hsparse = {'indices': sparse['indices'], 'values': (sparse_norm * (1 - alpha)).astype(np.float32).tolist()}
    
    return hdense, hsparse


def predict(prefix: str, top_k: int = None, alpha: float = None, 
            popularity_weight: float = None, use_prefix_filter: bool = True):
    """
    Predict top-k query completions for a given prefix
    
    Args:
        prefix: Search prefix (e.g., "nike sho")
        top_k: Number of predictions to return (default from config)
        alpha: Dense/sparse balance (default from config)
        popularity_weight: Weight for popularity boost (default from config)
        use_prefix_filter: Use 3-char prefix filtering for speed
    
    Returns:
        List of predicted queries
    """
    # Use default config values if not specified
    top_k = top_k or config['top_k']
    alpha = alpha or config['alpha']
    popularity_weight = popularity_weight or config['popularity_weight']
    
    # Optional: Pre-filter by prefix (3-char matching)
    if use_prefix_filter and len(prefix) >= 3:
        prefix_lower = prefix.lower()[:3]
        mask = df_pool_enhanced['prefix_3char'] == prefix_lower
        filtered_indices = mask[mask].index.to_numpy()
        
        if len(filtered_indices) == 0:
            use_prefix_filter = False  # Fall back to full search
    else:
        use_prefix_filter = False
    
    # Get dense embedding and apply PCA
    prefix_dense = model.encode([prefix])[0]
    prefix_dense = pca.transform([prefix_dense])[0].astype(np.float32)
    
    # Get sparse representation
    prefix_sparse = bm25.transform_query(prefix)
    
    # Normalize for hybrid scoring
    hdense, hsparse = hybrid_score_norm(prefix_dense.tolist(), prefix_sparse, alpha=alpha)
    
    if use_prefix_filter and len(filtered_indices) > 0:
        # Filtered search
        filtered_embeddings = pool_dense_embeddings[filtered_indices]
        filtered_sparse = pool_sparse_matrix[filtered_indices]
        filtered_popularity = df_pool_enhanced.iloc[filtered_indices]['popularity_score'].values
        
        # Dense similarity (manual computation for filtered subset)
        dense_scores = np.dot(filtered_embeddings, hdense)
        
        # Sparse similarity
        prefix_sparse_vec = csr_matrix(
            (hsparse['values'], (np.zeros(len(hsparse['indices']), dtype=int), hsparse['indices'])), 
            shape=(1, bm25.n_features)
        )
        sparse_scores = filtered_sparse.dot(prefix_sparse_vec.T).toarray().flatten()
        
        # Combine scores
        final_scores = combine_scores_fast(dense_scores, sparse_scores, filtered_popularity, popularity_weight)
        
        # Get top-k
        if len(final_scores) <= top_k:
            top_local_indices = np.argsort(final_scores)[::-1]
        else:
            top_local_indices = np.argpartition(final_scores, -top_k)[-top_k:]
            top_local_indices = top_local_indices[np.argsort(final_scores[top_local_indices])][::-1]
        
        top_indices = filtered_indices[top_local_indices]
    else:
        # Full FAISS search
        dense_scores, dense_indices = faiss_index.search(hdense.reshape(1, -1), top_k * 2)
        dense_scores = dense_scores.flatten()
        dense_indices = dense_indices.flatten()
        
        # Sparse scores
        prefix_sparse_vec = csr_matrix(
            (hsparse['values'], (np.zeros(len(hsparse['indices']), dtype=int), hsparse['indices'])), 
            shape=(1, bm25.n_features)
        )
        sparse_scores = pool_sparse_matrix[dense_indices].dot(prefix_sparse_vec.T).toarray().flatten()
        
        # Popularity scores
        popularity_scores = df_pool_enhanced.iloc[dense_indices]['popularity_score'].values
        
        # Combine scores
        final_scores = combine_scores_fast(dense_scores, sparse_scores, popularity_scores, popularity_weight)
        
        # Get top-k
        top_local_indices = np.argpartition(final_scores, -top_k)[-top_k:]
        top_local_indices = top_local_indices[np.argsort(final_scores[top_local_indices])][::-1]
        top_indices = dense_indices[top_local_indices]
    
    return [pool_queries[idx] for idx in top_indices]


# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Auto-completion Inference')
    parser.add_argument('--query', type=str, required=True, help='Search prefix to complete')
    parser.add_argument('--top-k', type=int, default=10, help='Number of predictions (default: 10)')
    parser.add_argument('--alpha', type=float, default=None, help='Dense/sparse balance (default: from config)')
    parser.add_argument('--popularity-weight', type=float, default=None, help='Popularity weight (default: from config)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Query: '{args.query}'")
    print(f"{'='*60}\n")
    
    predictions = predict(
        args.query, 
        top_k=args.top_k,
        alpha=args.alpha,
        popularity_weight=args.popularity_weight
    )
    
    print("Top predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  {i:2d}. {pred}")
    print()