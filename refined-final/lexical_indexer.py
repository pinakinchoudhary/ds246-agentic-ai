"""
Lexical retrieval using BM25 and character n-grams.
Provides fast prefix and fuzzy matching.
"""
import pickle
import numpy as np
from typing import List, Tuple, Set
from collections import defaultdict
import logging
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
import re

from config import *
from data_loader import DataLoader, normalize_text

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class LexicalIndexer:
    """Handles BM25 and n-gram based lexical search."""
    
    def __init__(self, force_rebuild: bool = False):
        self.force_rebuild = force_rebuild
        self.bm25_index = None
        self.ngram_index = None
        self.queries = []
        self.normalized_queries = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.split()
    
    def _get_character_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Get character n-grams from text."""
        text = text.replace(" ", "")  # Remove spaces for character ngrams
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def build_bm25_index(self, pool_df, force_rebuild: bool = False):
        """Build BM25 index for token-based search."""
        
        # Check cache
        if not force_rebuild and not self.force_rebuild and BM25_INDEX_PATH.exists():
            logger.info(f"Loading cached BM25 index from {BM25_INDEX_PATH}")
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.bm25_index = data["bm25_index"]
                self.queries = data["queries"]
                self.normalized_queries = data["normalized_queries"]
            logger.info(f"Loaded BM25 index for {len(self.queries):,} queries")
            return self.bm25_index
        
        # Build fresh index
        logger.info(f"Building BM25 index for {len(pool_df):,} queries...")
        
        self.queries = pool_df["query"].tolist()
        self.normalized_queries = [normalize_text(q) for q in self.queries]
        
        # Tokenize all queries
        tokenized_corpus = [self._tokenize(q) for q in self.normalized_queries]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
        
        # Save to disk
        logger.info(f"Saving BM25 index to {BM25_INDEX_PATH}")
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25_index": self.bm25_index,
                "queries": self.queries,
                "normalized_queries": self.normalized_queries
            }, f)
        
        logger.info(f"BM25 index built and saved")
        return self.bm25_index
    
    def build_ngram_index(self, pool_df, force_rebuild: bool = False):
        """Build character trigram index for fuzzy matching."""
        
        # Check cache
        if not force_rebuild and not self.force_rebuild and NGRAM_INDEX_PATH.exists():
            logger.info(f"Loading cached n-gram index from {NGRAM_INDEX_PATH}")
            with open(NGRAM_INDEX_PATH, "rb") as f:
                self.ngram_index = pickle.load(f)
            logger.info(f"Loaded n-gram index")
            return self.ngram_index
        
        # Build fresh index
        logger.info(f"Building trigram index...")
        
        if not self.queries:  # If not already loaded
            self.queries = pool_df["query"].tolist()
            self.normalized_queries = [normalize_text(q) for q in self.queries]
        
        # Build inverted index: trigram -> set of query indices
        ngram_to_indices = defaultdict(set)
        
        for idx, query in enumerate(self.normalized_queries):
            trigrams = self._get_character_ngrams(query, n=3)
            for trigram in trigrams:
                ngram_to_indices[trigram].add(idx)
        
        self.ngram_index = dict(ngram_to_indices)
        
        # Save to disk
        logger.info(f"Saving n-gram index to {NGRAM_INDEX_PATH}")
        with open(NGRAM_INDEX_PATH, "wb") as f:
            pickle.dump(self.ngram_index, f)
        
        logger.info(f"N-gram index built with {len(self.ngram_index):,} trigrams")
        return self.ngram_index
    
    def load_indices(self):
        """Load or build all indices."""
        loader = DataLoader()
        pool_df = loader.load_query_pool()
        
        self.build_bm25_index(pool_df)
        self.build_ngram_index(pool_df)
    
    def search_bm25(self, prefix: str, k: int = BM25_TOP_K) -> Tuple[List[int], List[float]]:
        """Search using BM25 for token overlap."""
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded")
        
        normalized_prefix = normalize_text(prefix)
        tokenized_query = self._tokenize(normalized_prefix)
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()
    
    def search_ngram(self, prefix: str, k: int = NGRAM_TOP_K) -> Tuple[List[int], List[float]]:
        """Search using trigram overlap."""
        if self.ngram_index is None:
            raise ValueError("N-gram index not loaded")
        
        normalized_prefix = normalize_text(prefix)
        
        # Get trigrams from prefix
        prefix_trigrams = self._get_character_ngrams(normalized_prefix, n=3)
        
        # Find queries with overlapping trigrams
        candidate_indices = set()
        for trigram in prefix_trigrams:
            if trigram in self.ngram_index:
                candidate_indices.update(self.ngram_index[trigram])
        
        if not candidate_indices:
            return [], []
        
        # Score by trigram overlap
        scores = []
        indices = []
        
        for idx in candidate_indices:
            query = self.normalized_queries[idx]
            query_trigrams = self._get_character_ngrams(query, n=3)
            
            # Jaccard similarity on trigrams
            intersection = len(prefix_trigrams & query_trigrams)
            union = len(prefix_trigrams | query_trigrams)
            score = intersection / union if union > 0 else 0
            
            scores.append(score)
            indices.append(idx)
        
        # Sort by score
        sorted_pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        top_k = sorted_pairs[:k]
        
        if not top_k:
            return [], []
        
        top_indices, top_scores = zip(*top_k)
        return list(top_indices), list(top_scores)
    
    def search_fuzzy(self, prefix: str, k: int = FUZZY_TOP_K) -> Tuple[List[int], List[float]]:
        """Fuzzy search using edit distance (expensive, use sparingly)."""
        if len(prefix) < MIN_PREFIX_LENGTH_FOR_FUZZY:
            return [], []
        
        normalized_prefix = normalize_text(prefix)
        
        # First get candidates from ngram search
        ngram_indices, _ = self.search_ngram(prefix, k=k*3)
        
        if not ngram_indices:
            return [], []
        
        # Score with fuzzy matching
        scores = []
        indices = []
        
        for idx in ngram_indices:
            query = self.normalized_queries[idx]
            score = fuzz.ratio(normalized_prefix, query) / 100.0
            
            if score >= FUZZY_THRESHOLD:
                scores.append(score)
                indices.append(idx)
        
        # Sort by score
        sorted_pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        top_k = sorted_pairs[:k]
        
        if not top_k:
            return [], []
        
        top_indices, top_scores = zip(*top_k)
        return list(top_indices), list(top_scores)
    
    def search_prefix_match(self, prefix: str, k: int = 20) -> List[int]:
        """Exact prefix matching."""
        normalized_prefix = normalize_text(prefix)
        
        matching_indices = []
        for idx, query in enumerate(self.normalized_queries):
            if query.startswith(normalized_prefix):
                matching_indices.append(idx)
                if len(matching_indices) >= k:
                    break
        
        return matching_indices


def main():
    """Build lexical indices."""
    logger.info("="*80)
    logger.info("LEXICAL INDEX BUILDING")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    pool_df = loader.load_query_pool()
    
    # Build indices
    indexer = LexicalIndexer(force_rebuild=False)
    indexer.build_bm25_index(pool_df)
    indexer.build_ngram_index(pool_df)
    
    # Test searches
    test_prefixes = ["ifon 16", "blaek blaijer", "running sho"]
    
    print("\nTest Results:")
    for prefix in test_prefixes:
        print(f"\n{'='*60}")
        print(f"Prefix: '{prefix}'")
        print(f"{'='*60}")
        
        # BM25 search
        bm25_indices, bm25_scores = indexer.search_bm25(prefix, k=5)
        print("\nBM25 Top 5:")
        for i, (idx, score) in enumerate(zip(bm25_indices, bm25_scores)):
            print(f"  {i+1}. {indexer.queries[idx]} (score: {score:.4f})")
        
        # N-gram search
        ngram_indices, ngram_scores = indexer.search_ngram(prefix, k=5)
        print("\nN-gram Top 5:")
        for i, (idx, score) in enumerate(zip(ngram_indices, ngram_scores)):
            print(f"  {i+1}. {indexer.queries[idx]} (score: {score:.4f})")
        
        # Fuzzy search
        fuzzy_indices, fuzzy_scores = indexer.search_fuzzy(prefix, k=5)
        print("\nFuzzy Top 5:")
        for i, (idx, score) in enumerate(zip(fuzzy_indices, fuzzy_scores)):
            print(f"  {i+1}. {indexer.queries[idx]} (score: {score:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info("LEXICAL INDEX BUILDING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()