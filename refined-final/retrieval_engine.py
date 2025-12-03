"""
Main retrieval engine for autocomplete.
Combines semantic, lexical retrieval with hardcoded scoring.
"""
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz

from config import *
from embedding_generator import EmbeddingGenerator
from lexical_indexer import LexicalIndexer
from data_loader import DataLoader, normalize_text

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Fast autocomplete retrieval with hardcoded scoring."""
    
    def __init__(self, use_cache: bool = ENABLE_CACHE):
        logger.info("Initializing Retrieval Engine...")
        
        # Load all components
        self.embedder = EmbeddingGenerator()
        self.lexical = LexicalIndexer()
        
        logger.info("Loading embeddings and indices...")
        self.embeddings, self.metadata, self.faiss_index = self.embedder.load_embeddings_and_index()
        self.lexical.load_indices()
        
        # Load query features
        logger.info("Loading query features...")
        loader = DataLoader()
        self.query_features_map = loader.get_query_to_features_map()
        
        # Cache
        self.use_cache = use_cache
        self.cache = {}
        
        # Popular queries for short prefixes
        self._build_popular_queries()
        
        logger.info("Retrieval Engine ready!")
    
    def _build_popular_queries(self):
        """Build list of most popular queries for short prefix handling."""
        logger.info("Building popular query list...")
        
        # Get queries sorted by popularity
        queries_with_scores = []
        for query in self.metadata["queries"]:
            feats = self.query_features_map.get(query, {})
            popularity = (
                feats.get("orders", 0) * 4 +
                feats.get("volume", 0) * 3 +
                feats.get("catalog_clicks", 0) * 2 +
                feats.get("catalog_views", 0)
            )
            queries_with_scores.append((query, popularity))
        
        # Sort by popularity
        queries_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.popular_queries = [q for q, _ in queries_with_scores[:SHORT_PREFIX_TOP_N]]
        logger.info(f"Built list of {len(self.popular_queries):,} popular queries")
    
    def _retrieve_candidates(self, prefix: str) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
        """
        Retrieve candidates using multiple methods.
        
        Returns:
            queries, semantic_scores, bm25_scores, ngram_scores, fuzzy_scores
        """
        candidates = {}  # query -> [sem, bm25, ngram, fuzzy]
        
        if PARALLEL_RETRIEVAL:
            # Run retrievals in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit tasks
                future_semantic = executor.submit(self._retrieve_semantic, prefix)
                future_bm25 = executor.submit(self._retrieve_bm25, prefix)
                future_ngram = executor.submit(self._retrieve_ngram, prefix)
                
                # Get results
                sem_results = future_semantic.result()
                bm25_results = future_bm25.result()
                ngram_results = future_ngram.result()
        else:
            # Sequential
            sem_results = self._retrieve_semantic(prefix)
            bm25_results = self._retrieve_bm25(prefix)
            ngram_results = self._retrieve_ngram(prefix)
        
        # Merge results
        for query, score in sem_results:
            if query not in candidates:
                candidates[query] = [0, 0, 0, 0]
            candidates[query][0] = score
        
        for query, score in bm25_results:
            if query not in candidates:
                candidates[query] = [0, 0, 0, 0]
            candidates[query][1] = score
        
        for query, score in ngram_results:
            if query not in candidates:
                candidates[query] = [0, 0, 0, 0]
            candidates[query][2] = score
        
        # Fuzzy only for longer prefixes
        if len(prefix) >= MIN_PREFIX_LENGTH_FOR_FUZZY:
            fuzzy_results = self._retrieve_fuzzy(prefix)
            for query, score in fuzzy_results:
                if query not in candidates:
                    candidates[query] = [0, 0, 0, 0]
                candidates[query][3] = score
        
        # Limit total candidates
        if len(candidates) > MAX_CANDIDATES:
            # Keep top by max score across methods
            candidates_list = [(q, max(scores)) for q, scores in candidates.items()]
            candidates_list.sort(key=lambda x: x[1], reverse=True)
            candidates_list = candidates_list[:MAX_CANDIDATES]
            
            # Rebuild dict
            kept_queries = {q for q, _ in candidates_list}
            candidates = {q: scores for q, scores in candidates.items() if q in kept_queries}
        
        # Convert to lists
        queries = list(candidates.keys())
        scores = list(candidates.values())
        
        semantic_scores = [s[0] for s in scores]
        bm25_scores = [s[1] for s in scores]
        ngram_scores = [s[2] for s in scores]
        fuzzy_scores = [s[3] for s in scores]
        
        return queries, semantic_scores, bm25_scores, ngram_scores, fuzzy_scores
    
    def _retrieve_semantic(self, prefix: str) -> List[Tuple[str, float]]:
        """Semantic retrieval."""
        prefix_emb = self.embedder.encode_queries([prefix])
        distances, indices = self.embedder.search(prefix_emb, k=SEMANTIC_TOP_K)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            query = self.metadata["queries"][idx]
            results.append((query, float(dist)))
        
        return results
    
    def _retrieve_bm25(self, prefix: str) -> List[Tuple[str, float]]:
        """BM25 retrieval."""
        indices, scores = self.lexical.search_bm25(prefix, k=BM25_TOP_K)
        
        results = []
        for idx, score in zip(indices, scores):
            query = self.lexical.queries[idx]
            results.append((query, float(score)))
        
        return results
    
    def _retrieve_ngram(self, prefix: str) -> List[Tuple[str, float]]:
        """N-gram retrieval."""
        indices, scores = self.lexical.search_ngram(prefix, k=NGRAM_TOP_K)
        
        results = []
        for idx, score in zip(indices, scores):
            query = self.lexical.queries[idx]
            results.append((query, float(score)))
        
        return results
    
    def _retrieve_fuzzy(self, prefix: str) -> List[Tuple[str, float]]:
        """Fuzzy retrieval."""
        indices, scores = self.lexical.search_fuzzy(prefix, k=FUZZY_TOP_K)
        
        results = []
        for idx, score in zip(indices, scores):
            query = self.lexical.queries[idx]
            results.append((query, float(score)))
        
        return results
    
    def _compute_scores(
        self, 
        prefix: str, 
        queries: List[str], 
        sem_scores: List[float],
        bm25_scores: List[float],
        ngram_scores: List[float],
        fuzzy_scores: List[float]
    ) -> np.ndarray:
        """Compute final scores using hardcoded weights."""
        scores = []
        norm_prefix = normalize_text(prefix)
        prefix_len = len(prefix)
        
        for i, query in enumerate(queries):
            norm_query = normalize_text(query)
            
            # Base retrieval scores (already normalized)
            sem_score = sem_scores[i]
            bm25_score = bm25_scores[i] / 10.0 if bm25_scores[i] > 0 else 0  # Normalize BM25
            ngram_score = ngram_scores[i]
            fuzzy_score = fuzzy_scores[i]
            
            # String matching features
            exact_match = 1.0 if norm_prefix == norm_query else 0.0
            starts_with = 1.0 if norm_query.startswith(norm_prefix) else 0.0
            contains = 1.0 if norm_prefix in norm_query else 0.0
            
            # Fuzzy string similarity
            ratio = fuzz.ratio(norm_prefix, norm_query) / 100.0
            partial_ratio = fuzz.partial_ratio(norm_prefix, norm_query) / 100.0
            
            # Length features
            len_diff = abs(len(norm_query) - prefix_len)
            len_ratio = min(prefix_len, len(norm_query)) / max(prefix_len, len(norm_query)) if max(prefix_len, len(norm_query)) > 0 else 0
            
            # Popularity features (log-transformed and normalized)
            query_feats = self.query_features_map.get(query, {})
            orders = np.log1p(query_feats.get("orders", 0)) / 10.0
            volume = np.log1p(query_feats.get("volume", 0)) / 15.0
            clicks = np.log1p(query_feats.get("catalog_clicks", 0)) / 12.0
            views = np.log1p(query_feats.get("catalog_views", 0)) / 12.0
            
            # Hardcoded weighted score
            # Tuned weights based on feature importance from previous LightGBM runs
            score = (
                exact_match * 50.0 +           # Exact match is king
                starts_with * 20.0 +            # Prefix match is very important
                sem_score * 30.0 +              # Semantic similarity
                partial_ratio * 25.0 +          # Partial fuzzy match
                bm25_score * 20.0 +             # BM25 score
                orders * 15.0 +                 # Order popularity
                ratio * 12.0 +                  # Full fuzzy ratio
                ngram_score * 10.0 +            # N-gram similarity
                volume * 15.0 +                  # Search volume
                clicks * 15.0 +                  # Catalog clicks
                contains * 5.0 +                # Contains prefix
                views * 10.0 +                   # Catalog views
                len_ratio * 0.0 +               # Length similarity
                fuzzy_score * 20.0 -             # Fuzzy retrieval score
                len_diff * 0.0                # Penalize length difference
            )
            
            scores.append(score)
        
        return np.array(scores)
    
    def _handle_short_prefix(self, prefix: str) -> List[str]:
        """Handle very short prefixes with popularity-based results."""
        norm_prefix = normalize_text(prefix)
        
        # Filter popular queries by prefix match
        results = []
        for query in self.popular_queries:
            norm_query = normalize_text(query)
            if norm_query.startswith(norm_prefix):
                results.append(query)
                if len(results) >= FINAL_TOP_K:
                    break
        
        return results
    
    def retrieve(self, prefix: str, return_scores: bool = False) -> List[str]:
        """
        Main retrieval method.
        
        Args:
            prefix: User input prefix
            return_scores: If True, return (queries, scores) tuple
        
        Returns:
            List of top-k query completions (or tuple if return_scores=True)
        """
        # Check cache
        if self.use_cache and prefix in self.cache:
            return self.cache[prefix]
        
        start_time = time.time()
        
        # Handle very short prefixes
        if len(prefix) <= SHORT_PREFIX_THRESHOLD:
            results = self._handle_short_prefix(prefix)
            if results:
                if self.use_cache:
                    self.cache[prefix] = results
                return results
        
        # Retrieve candidates
        queries, sem_scores, bm25_scores, ngram_scores, fuzzy_scores = self._retrieve_candidates(prefix)
        
        if not queries:
            return []
        
        # Compute simple scores with hardcoded weights
        predictions = self._compute_scores(prefix, queries, sem_scores, bm25_scores, ngram_scores, fuzzy_scores)
        
        # Sort by score
        ranked_indices = np.argsort(predictions)[::-1]
        
        # Get top k
        top_indices = ranked_indices[:FINAL_TOP_K]
        top_queries = [queries[i] for i in top_indices]
        top_scores = [predictions[i] for i in top_indices]
        
        # Cache
        if self.use_cache:
            self.cache[prefix] = top_queries
            
            # Limit cache size
            if len(self.cache) > CACHE_SIZE:
                # Remove oldest (simple FIFO)
                self.cache.pop(next(iter(self.cache)))
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Retrieved {len(top_queries)} results in {elapsed:.2f}ms")
        
        if return_scores:
            return top_queries, top_scores
        return top_queries
    
    def batch_retrieve(self, prefixes: List[str]) -> List[List[str]]:
        """Retrieve for multiple prefixes."""
        results = []
        for prefix in prefixes:
            results.append(self.retrieve(prefix))
        return results


def main():
    """Test retrieval engine."""
    logger.info("="*80)
    logger.info("TESTING RETRIEVAL ENGINE")
    logger.info("="*80)
    
    # Initialize engine
    engine = RetrievalEngine()
    
    # Test queries
    test_prefixes = [
        "ifon 16",
        "blaek blaijer",
        "running sho",
        "samsu",
        "fire boat",
        "kids kh"
    ]
    
    print("\nTest Results:")
    print("="*80)
    
    for prefix in test_prefixes:
        print(f"\nPrefix: '{prefix}'")
        print("-"*80)
        
        start = time.time()
        results, scores = engine.retrieve(prefix, return_scores=True)
        elapsed = (time.time() - start) * 1000
        
        print(f"Top {len(results)} completions (in {elapsed:.2f}ms):")
        for i, (query, score) in enumerate(zip(results, scores), 1):
            print(f"  {i:2d}. {query:40s} (score: {score:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
