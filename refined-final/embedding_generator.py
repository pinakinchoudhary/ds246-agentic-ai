"""
Embedding generation using GPU with caching.
Generates and saves embeddings for the entire query pool.
"""
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from typing import List, Tuple
import faiss

from config import *
from data_loader import DataLoader, normalize_text

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation with GPU acceleration and caching."""
    
    def __init__(self, force_regenerate: bool = False):
        self.force_regenerate = force_regenerate
        self.device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        self.model.max_seq_length = 128  # Limit for speed
        
        # Check if embeddings exist
        self.embeddings = None
        self.metadata = None
        self.faiss_index = None
        
    def generate_pool_embeddings(self, pool_df) -> Tuple[np.ndarray, dict]:
        """Generate embeddings for entire query pool."""
        
        # Check cache
        if not self.force_regenerate and POOL_EMBEDDINGS_PATH.exists():
            logger.info(f"Loading cached embeddings from {POOL_EMBEDDINGS_PATH}")
            embeddings = np.load(POOL_EMBEDDINGS_PATH)
            with open(POOL_METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
            logger.info(f"Loaded {len(embeddings):,} cached embeddings")
            return embeddings, metadata
        
        # Generate fresh embeddings
        logger.info(f"Generating embeddings for {len(pool_df):,} queries...")
        
        queries = pool_df["query"].tolist()
        normalized_queries = [normalize_text(q) for q in queries]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            normalized_queries,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        # Create metadata
        metadata = {
            "queries": queries,  # Original queries
            "normalized_queries": normalized_queries,
            "size": len(queries),
            "dimension": embeddings.shape[1]
        }
        
        # Save to disk
        logger.info(f"Saving embeddings to {POOL_EMBEDDINGS_PATH}")
        np.save(POOL_EMBEDDINGS_PATH, embeddings)
        with open(POOL_METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Generated and saved {len(embeddings):,} embeddings")
        return embeddings, metadata
    
    def build_faiss_index(self, embeddings: np.ndarray, force_rebuild: bool = False):
        """Build FAISS HNSW index for fast ANN search."""
        
        # Check cache
        if not force_rebuild and FAISS_INDEX_PATH.exists():
            logger.info(f"Loading cached FAISS index from {FAISS_INDEX_PATH}")
            
            if USE_GPU and torch.cuda.is_available():
                # Try to use GPU index
                try:
                    cpu_index = faiss.read_index(str(FAISS_INDEX_PATH))
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    logger.info("Loaded FAISS index to GPU")
                except Exception as e:
                    logger.warning(f"Could not load to GPU: {e}, using CPU")
                    self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            else:
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            
            return self.faiss_index
        
        # Build fresh index
        logger.info(f"Building FAISS HNSW index...")
        d = embeddings.shape[1]
        
        # HNSW index for best recall/latency
        index = faiss.IndexHNSWFlat(d, FAISS_HNSW_M)
        index.hnsw.efConstruction = FAISS_EF_CONSTRUCTION
        
        logger.info(f"Adding {len(embeddings):,} vectors to index...")
        index.add(embeddings)
        index.hnsw.efSearch = FAISS_EF_SEARCH
        
        # Save CPU version
        logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}")
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        
        # Load to GPU if available
        if USE_GPU and torch.cuda.is_available():
            try:
                logger.info("Moving FAISS index to GPU...")
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index loaded to GPU")
            except Exception as e:
                logger.warning(f"Could not load to GPU: {e}, using CPU")
                self.faiss_index = index
        else:
            self.faiss_index = index
        
        return self.faiss_index
    
    def load_embeddings_and_index(self) -> Tuple[np.ndarray, dict, faiss.Index]:
        """Load or generate embeddings and build FAISS index."""
        if self.embeddings is None or self.metadata is None:
            loader = DataLoader()
            pool_df = loader.load_query_pool()
            self.embeddings, self.metadata = self.generate_pool_embeddings(pool_df)
        
        if self.faiss_index is None:
            self.faiss_index = self.build_faiss_index(self.embeddings)
        
        return self.embeddings, self.metadata, self.faiss_index
    
    def encode_queries(self, queries: List[str], normalize: bool = True) -> np.ndarray:
        """Encode a list of queries to embeddings."""
        if normalize:
            queries = [normalize_text(q) for q in queries]
        
        embeddings = self.model.encode(
            queries,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def search(self, query_embeddings: np.ndarray, k: int = SEMANTIC_TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index for nearest neighbors."""
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded. Call load_embeddings_and_index() first.")
        
        # Ensure query embeddings are normalized
        faiss.normalize_L2(query_embeddings)
        
        # Search
        distances, indices = self.faiss_index.search(query_embeddings, k)
        return distances, indices


def main():
    """Generate embeddings and build FAISS index."""
    logger.info("="*80)
    logger.info("EMBEDDING GENERATION PIPELINE")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    pool_df = loader.load_query_pool()
    
    # Generate embeddings
    generator = EmbeddingGenerator(force_regenerate=False)
    embeddings, metadata = generator.generate_pool_embeddings(pool_df)
    
    logger.info(f"\nEmbedding stats:")
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Memory: {embeddings.nbytes / 1e6:.2f} MB")
    logger.info(f"  Dimension: {metadata['dimension']}")
    
    # Build FAISS index
    faiss_index = generator.build_faiss_index(embeddings, force_rebuild=False)
    
    logger.info(f"\nFAISS index built:")
    logger.info(f"  Total vectors: {faiss_index.ntotal:,}")
    
    # Test search
    logger.info("\nTesting semantic search...")
    test_queries = ["iphone 16", "black blazer", "running shoes"]
    test_embeddings = generator.encode_queries(test_queries)
    distances, indices = generator.search(test_embeddings, k=5)
    
    print("\nTest Results:")
    for i, query in enumerate(test_queries):
        print(f"\nQuery: '{query}'")
        print("Top 5 matches:")
        for j, (dist, idx) in enumerate(zip(distances[i], indices[i])):
            result_query = metadata["queries"][idx]
            print(f"  {j+1}. {result_query} (similarity: {dist:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()