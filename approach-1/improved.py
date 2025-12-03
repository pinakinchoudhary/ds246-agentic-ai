import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from typing import List, Callable, Optional, Dict
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm
import faiss
from numba import jit

# Load spacy for tokenization
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

SparseVector = Dict[str, List]
Array = List[float]


class BM25:
    """Implementation of OKapi BM25 with HashingVectorizer - Optimized"""

    def __init__(self, tokenizer: Callable[[str], List[str]], n_features=2 ** 14, b=0.75, k1=1.6):
        self.ndocs: int = 0
        self.n_features: int = n_features
        self.doc_freq: Array = []
        self.avgdl: Optional[float] = None
        self._tokenizer: Callable[[str], List[str]] = tokenizer
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            token_pattern=None,
            tokenizer=tokenizer,
            norm=None,
            alternate_sign=False,
            binary=True
        )
        self.b: float = b
        self.k1: float = k1

    def fit(self, corpus: List[str]) -> "BM25":
        """Fit IDF to documents"""
        X = self._vectorizer.transform(corpus)
        self.avgdl = X.sum(1).mean()
        self.ndocs = X.shape[0]
        self.doc_freq = (
            self._vectorizer
            .transform(corpus)
            .sum(axis=0)
            .A1
        )
        return self

    def transform_doc(self, doc: str) -> SparseVector:
        """Normalize document for BM25 scoring"""
        doc_tf = self._vectorizer.transform([doc])
        norm_doc_tf = self._norm_doc_tf(doc_tf)
        return {'indices': [int(x) for x in doc_tf.indices], 'values': norm_doc_tf}

    def transform_query(self, query: str) -> SparseVector:
        """Normalize query for BM25 scoring"""
        query_tf = self._vectorizer.transform([query])
        indices, values = self._norm_query_tf(query_tf)
        return {'indices': [int(x) for x in indices], 'values': values}
    
    def transform_docs_batch(self, docs: List[str]) -> csr_matrix:
        """Batch transform documents to sparse matrix"""
        doc_tfs = self._vectorizer.transform(docs)
        b, k1, avgdl = self.b, self.k1, self.avgdl
        
        # Normalize each document
        normalized_data = []
        for i in range(doc_tfs.shape[0]):
            row = doc_tfs.getrow(i)
            tf = row.data
            dl = tf.sum()
            norm_tf = tf / (k1 * (1.0 - b + b * (dl / avgdl)) + tf)
            normalized_data.append(norm_tf)
        
        # Reconstruct sparse matrix
        normalized_matrix = csr_matrix((
            np.concatenate(normalized_data),
            doc_tfs.indices,
            doc_tfs.indptr
        ), shape=doc_tfs.shape)
        
        return normalized_matrix

    def _norm_doc_tf(self, doc_tf) -> np.ndarray:
        """Calculate BM25 normalized document term-frequencies"""
        b, k1, avgdl = self.b, self.k1, self.avgdl
        tf = doc_tf.data
        dl = tf.sum()
        norm_tf = tf / (k1 * (1.0 - b + b * (dl / avgdl)) + tf)
        return norm_tf

    def _norm_query_tf(self, query_tf):
        """Calculate BM25 normalized query term-frequencies"""
        df = self.doc_freq[query_tf.indices]
        idf = np.log((self.ndocs + 1) / (df + 0.5))
        norm_query_tf = idf / idf.sum()
        return query_tf.indices, norm_query_tf


def tokenizer(text):
    """Tokenize text using spacy"""
    return [token.text.lower() for token in nlp(text)]


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


# Loading the Data from Hugging Face Hub
print("Loading datasets...")
train_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="train_data/*.parquet")
test_prefixes_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="test_prefixes_data/*.parquet")
query_features = load_dataset("123tushar/Dice_Challenge_2025", data_files="query_features/*.parquet")
pool = load_dataset("123tushar/Dice_Challenge_2025", data_files="pool/*.parquet")

# Reading the Data to pandas DF
df_train = train_data['train'].to_pandas()
df_test = test_prefixes_data['train'].to_pandas()
df_query_features = query_features['train'].to_pandas()
df_pool = pool['train'].to_pandas()

print(f"Train data: {len(df_train)} prefix-query pairs")
print(f"Test prefixes: {len(df_test)} prefixes")
print(f"Query pool: {len(df_pool)} queries")
print(f"Query features: {len(df_query_features)} queries")

# Merge query features with pool for enhanced retrieval
df_pool_enhanced = df_pool.merge(df_query_features, on='query', how='left')
df_pool_enhanced.fillna(0, inplace=True)

# Create prefix lookup for fast filtering
print("\nBuilding prefix lookup index...")
df_pool_enhanced['query_lower'] = df_pool_enhanced['query'].str.lower()
df_pool_enhanced['prefix_3char'] = df_pool_enhanced['query_lower'].str[:3]

# Initialize BM25 on the query pool (REDUCED FEATURES)
print("\nFitting BM25 on query pool...")
bm25 = BM25(tokenizer=tokenizer, n_features=2**14, b=0.75, k1=1.6)
bm25.fit(df_pool['query'].tolist())

# Initialize Sentence Transformer for dense embeddings
print("Loading Sentence Transformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Pre-compute dense embeddings for the entire pool
print("Computing dense embeddings for query pool...")
pool_queries = df_pool['query'].tolist()
pool_dense_embeddings = model.encode(pool_queries, show_progress_bar=True, batch_size=64)

# Apply PCA for dimensionality reduction
print("Applying PCA dimensionality reduction...")
original_dim = pool_dense_embeddings.shape[1]
target_dim = 128
pca = PCA(n_components=target_dim)
pool_dense_embeddings = pca.fit_transform(pool_dense_embeddings).astype(np.float32)
print(f"Reduced embeddings from {original_dim}D to {target_dim}D")

# Build FAISS index for fast approximate nearest neighbor search
print("Building FAISS index...")
faiss_index = faiss.IndexFlatIP(target_dim)  # Inner product for cosine similarity
faiss_index.add(pool_dense_embeddings)
print(f"FAISS index built with {faiss_index.ntotal} vectors")

# Pre-compute sparse embeddings in batches and store as sparse matrix
print("Computing sparse embeddings for query pool (in batches)...")
batch_size = 10000
sparse_matrices = []

for i in tqdm(range(0, len(pool_queries), batch_size)):
    batch = pool_queries[i:i+batch_size]
    sparse_batch = bm25.transform_docs_batch(batch)
    sparse_matrices.append(sparse_batch)

# Concatenate all sparse matrices - FIX: Use float32 instead of float16
pool_sparse_matrix = vstack(sparse_matrices).astype(np.float32)
print(f"Sparse matrix shape: {pool_sparse_matrix.shape}, Memory: {pool_sparse_matrix.data.nbytes / 1e9:.2f} GB")

# Sparsify further by removing low-value terms
threshold = 0.01
pool_sparse_matrix.data[pool_sparse_matrix.data < threshold] = 0
pool_sparse_matrix.eliminate_zeros()
print(f"After sparsification, Memory: {pool_sparse_matrix.data.nbytes / 1e9:.2f} GB")

# Clear temporary data
del sparse_matrices

# Normalize query features for boosting
feature_cols = ['catalog_clicks', 'orders', 'volume', 'catalog_views']
for col in feature_cols:
    if col in df_pool_enhanced.columns:
        max_val = df_pool_enhanced[col].max()
        if max_val > 0:
            df_pool_enhanced[f'{col}_norm'] = df_pool_enhanced[col] / max_val
        else:
            df_pool_enhanced[f'{col}_norm'] = 0

# Calculate popularity score
df_pool_enhanced['popularity_score'] = (
    df_pool_enhanced.get('orders_norm', 0) * 0.4 +
    df_pool_enhanced.get('catalog_clicks_norm', 0) * 0.3 +
    df_pool_enhanced.get('volume_norm', 0) * 0.2 +
    df_pool_enhanced.get('catalog_views_norm', 0) * 0.1
).astype(np.float32)


def retrieve_candidates(prefix: str, top_k: int = 100, alpha: float = 0.6, 
                       popularity_weight: float = 0.15, use_prefix_filter: bool = True):
    """Retrieve top-k candidate queries using optimized hybrid search with FAISS"""
    
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
    
    if use_prefix_filter:
        # Filtered FAISS search
        filtered_embeddings = pool_dense_embeddings[filtered_indices]
        filtered_sparse = pool_sparse_matrix[filtered_indices]
        filtered_popularity = df_pool_enhanced.iloc[filtered_indices]['popularity_score'].values
        
        # Dense similarity using FAISS on filtered subset
        dense_scores, _ = faiss_index.search(hdense.reshape(1, -1), min(top_k * 3, len(filtered_indices)))
        dense_scores = dense_scores.flatten()
        
        # Sparse similarity
        prefix_sparse_vec = csr_matrix((hsparse['values'], (np.zeros(len(hsparse['indices']), dtype=int), hsparse['indices'])), 
                                        shape=(1, bm25.n_features))
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
        # Full FAISS search (faster than brute force)
        dense_scores, dense_indices = faiss_index.search(hdense.reshape(1, -1), top_k * 2)
        dense_scores = dense_scores.flatten()
        dense_indices = dense_indices.flatten()
        
        # Only compute sparse for top dense candidates
        prefix_sparse_vec = csr_matrix((hsparse['values'], (np.zeros(len(hsparse['indices']), dtype=int), hsparse['indices'])), 
                                        shape=(1, bm25.n_features))
        sparse_scores = pool_sparse_matrix[dense_indices].dot(prefix_sparse_vec.T).toarray().flatten()
        
        # Get popularity scores
        popularity_scores = df_pool_enhanced.iloc[dense_indices]['popularity_score'].values
        
        # Combine scores
        final_scores = combine_scores_fast(dense_scores, sparse_scores, popularity_scores, popularity_weight)
        
        # Get top-k from candidates
        top_local_indices = np.argpartition(final_scores, -top_k)[-top_k:]
        top_local_indices = top_local_indices[np.argsort(final_scores[top_local_indices])][::-1]
        top_indices = dense_indices[top_local_indices]
    
    return [pool_queries[idx] for idx in top_indices]


def retrieve_candidates_batch(prefixes: List[str], top_k: int = 100, alpha: float = 0.6, 
                              popularity_weight: float = 0.15):
    """Batch retrieve candidates for multiple prefixes"""
    results = []
    
    # Batch encode all prefixes
    print("Batch encoding prefixes...")
    prefix_embeddings = model.encode(prefixes, show_progress_bar=True, batch_size=64)
    prefix_embeddings = pca.transform(prefix_embeddings).astype(np.float32)
    
    print("Retrieving candidates...")
    for i, prefix in enumerate(tqdm(prefixes)):
        prefix_dense = prefix_embeddings[i]
        prefix_sparse = bm25.transform_query(prefix)
        
        # Normalize
        hdense, hsparse = hybrid_score_norm(prefix_dense.tolist(), prefix_sparse, alpha=alpha)
        
        # FAISS search
        dense_scores, dense_indices = faiss_index.search(hdense.reshape(1, -1), top_k * 2)
        dense_scores = dense_scores.flatten()
        dense_indices = dense_indices.flatten()
        
        # Sparse scores
        prefix_sparse_vec = csr_matrix((hsparse['values'], (np.zeros(len(hsparse['indices']), dtype=int), hsparse['indices'])), 
                                        shape=(1, bm25.n_features))
        sparse_scores = pool_sparse_matrix[dense_indices].dot(prefix_sparse_vec.T).toarray().flatten()
        
        # Popularity
        popularity_scores = df_pool_enhanced.iloc[dense_indices]['popularity_score'].values
        
        # Combine
        final_scores = combine_scores_fast(dense_scores, sparse_scores, popularity_scores, popularity_weight)
        
        # Top-k
        top_local_indices = np.argpartition(final_scores, -top_k)[-top_k:]
        top_local_indices = top_local_indices[np.argsort(final_scores[top_local_indices])][::-1]
        top_indices = dense_indices[top_local_indices]
        
        results.append({
            'prefix': prefix,
            'candidates': [pool_queries[idx] for idx in top_indices]
        })
    
    return results


# Run inference on test prefixes (BATCH MODE)
print("\nGenerating predictions for test prefixes...")

# Process first 10 as a test
test_prefixes_sample = df_test['prefix'].tolist()[:10]
results = retrieve_candidates_batch(test_prefixes_sample, top_k=100, alpha=0.6, popularity_weight=0.15)

print(f"\nSample results:")
for i in range(min(3, len(results))):
    print(f"\nPrefix: '{results[i]['prefix']}'")
    print(f"Top 5 candidates: {results[i]['candidates'][:5]}")

# Save results
results_df = pd.DataFrame(results)
print(f"\nGenerated {len(results_df)} predictions")

import pickle
import numpy as np
import faiss
import joblib
from pathlib import Path

# Create directory for model artifacts
model_dir = Path('model_artifacts')
model_dir.mkdir(exist_ok=True)

print("Saving model components...")

# 1. Save BM25 model
print("Saving BM25 model...")
with open(model_dir / 'bm25_model.pkl', 'wb') as f:
    pickle.dump(bm25, f)

# 2. Save PCA model
print("Saving PCA model...")
joblib.dump(pca, model_dir / 'pca_model.pkl')

# 3. Save Sentence Transformer model path (we'll reload it during inference)
print("Saving model configuration...")
model_config = {
    'sentence_transformer_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'target_dim': target_dim,
    'alpha': 0.6,
    'popularity_weight': 0.15,
    'top_k': 100
}
with open(model_dir / 'model_config.pkl', 'wb') as f:
    pickle.dump(model_config, f)

# 4. Save FAISS index
print("Saving FAISS index...")
faiss.write_index(faiss_index, str(model_dir / 'faiss_index.bin'))

# 5. Save pool data and embeddings
print("Saving pool data...")
# Save the query pool
df_pool_enhanced[['query', 'query_lower', 'prefix_3char', 'popularity_score']].to_parquet(
    model_dir / 'pool_data.parquet'
)

# Save dense embeddings
np.save(model_dir / 'pool_dense_embeddings.npy', pool_dense_embeddings)

# 6. Save sparse matrix
print("Saving sparse embeddings...")
from scipy.sparse import save_npz
save_npz(model_dir / 'pool_sparse_matrix.npz', pool_sparse_matrix)

# 7. Save pool queries list for quick lookup
with open(model_dir / 'pool_queries.pkl', 'wb') as f:
    pickle.dump(pool_queries, f)

print(f"\n✅ All model components saved to '{model_dir}/' directory")
print("\nSaved files:")
for file in sorted(model_dir.iterdir()):
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"  - {file.name}: {size_mb:.2f} MB")