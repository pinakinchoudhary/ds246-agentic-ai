"""
Feature extraction for ranking model.
Computes features from prefix-query pairs for LightGBM.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from rapidfuzz import fuzz
from collections import Counter
import logging

from config import *
from data_loader import normalize_text

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features for ranking model."""
    
    def __init__(self, query_features_map: Dict[str, Dict]):
        """
        Args:
            query_features_map: Dict mapping query -> {orders, volume, clicks, views}
        """
        self.query_features_map = query_features_map
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return text.split()
    
    def _get_trigrams(self, text: str) -> set:
        """Get character trigrams."""
        text = text.replace(" ", "")
        if len(text) < 3:
            return {text}
        return {text[i:i+3] for i in range(len(text) - 3 + 1)}
    
    def extract_single(
        self,
        prefix: str,
        query: str,
        semantic_score: float = 0.0,
        bm25_score: float = 0.0,
        ngram_score: float = 0.0,
        fuzzy_score: float = 0.0
    ) -> Dict[str, float]:
        """
        Extract all features for a single prefix-query pair.
        
        Returns:
            Dictionary of feature_name -> value
        """
        features = {}
        
        # Normalize
        norm_prefix = normalize_text(prefix)
        norm_query = normalize_text(query)
        
        # Tokenize
        prefix_tokens = self._tokenize(norm_prefix)
        query_tokens = self._tokenize(norm_query)
        
        # Trigrams
        prefix_trigrams = self._get_trigrams(norm_prefix)
        query_trigrams = self._get_trigrams(norm_query)
        
        # === Retrieval scores ===
        features["semantic_cosine"] = semantic_score
        features["bm25_score"] = bm25_score
        features["ngram_score"] = ngram_score
        features["fuzzy_score"] = fuzzy_score
        
        # === String similarity features ===
        features["edit_distance_ratio"] = fuzz.ratio(norm_prefix, norm_query) / 100.0
        features["partial_ratio"] = fuzz.partial_ratio(norm_prefix, norm_query) / 100.0
        features["token_sort_ratio"] = fuzz.token_sort_ratio(norm_prefix, norm_query) / 100.0
        
        # === Prefix matching ===
        features["exact_prefix_match"] = float(norm_query.startswith(norm_prefix))
        features["prefix_in_query"] = float(norm_prefix in norm_query)
        
        # === Token overlap ===
        prefix_set = set(prefix_tokens)
        query_set = set(query_tokens)
        
        intersection = len(prefix_set & query_set)
        union = len(prefix_set | query_set)
        
        features["token_jaccard"] = intersection / union if union > 0 else 0
        features["token_overlap_count"] = intersection
        features["token_containment"] = intersection / len(prefix_set) if prefix_set else 0
        
        # === Trigram overlap ===
        trigram_intersection = len(prefix_trigrams & query_trigrams)
        trigram_union = len(prefix_trigrams | query_trigrams)
        
        features["trigram_jaccard"] = trigram_intersection / trigram_union if trigram_union > 0 else 0
        features["trigram_overlap_count"] = trigram_intersection
        
        # === Length features ===
        features["prefix_length"] = len(norm_prefix)
        features["query_length"] = len(norm_query)
        features["length_diff"] = abs(len(norm_query) - len(norm_prefix))
        features["length_ratio"] = len(norm_prefix) / len(norm_query) if len(norm_query) > 0 else 0
        
        features["prefix_token_count"] = len(prefix_tokens)
        features["query_token_count"] = len(query_tokens)
        features["token_count_diff"] = abs(len(query_tokens) - len(prefix_tokens))
        
        # === Position features ===
        if norm_prefix and norm_query:
            # Where does prefix appear in query
            try:
                pos = norm_query.index(norm_prefix)
                features["prefix_position"] = pos / len(norm_query)
            except ValueError:
                features["prefix_position"] = 1.0  # Not found
        else:
            features["prefix_position"] = 1.0
        
        # === Query quality/popularity features ===
        query_feats = self.query_features_map.get(query, {})
        
        orders = query_feats.get("orders", 0)
        volume = query_feats.get("volume", 0)
        clicks = query_feats.get("catalog_clicks", 0)
        views = query_feats.get("catalog_views", 0)
        
        # Log transform for heavy-tailed distributions
        features["log_orders"] = np.log1p(orders)
        features["log_volume"] = np.log1p(volume)
        features["log_clicks"] = np.log1p(clicks)
        features["log_views"] = np.log1p(views)
        
        # Raw values (will be normalized later)
        features["orders"] = orders
        features["volume"] = volume
        features["clicks"] = clicks
        features["views"] = views
        
        # Engagement ratios
        features["click_through_rate"] = clicks / volume if volume > 0 else 0
        features["order_rate"] = orders / volume if volume > 0 else 0
        features["views_per_click"] = views / clicks if clicks > 0 else 0
        
        # Combined popularity score
        features["popularity_score"] = (
            0.4 * np.log1p(orders) +
            0.3 * np.log1p(volume) +
            0.2 * np.log1p(clicks) +
            0.1 * np.log1p(views)
        )
        
        return features
    
    def extract_batch(
        self,
        prefixes: List[str],
        queries: List[str],
        semantic_scores: List[float],
        bm25_scores: List[float],
        ngram_scores: List[float],
        fuzzy_scores: List[float]
    ) -> pd.DataFrame:
        """
        Extract features for a batch of prefix-query pairs.
        
        Returns:
            DataFrame with one row per pair
        """
        feature_dicts = []
        
        for i in range(len(prefixes)):
            feats = self.extract_single(
                prefix=prefixes[i],
                query=queries[i],
                semantic_score=semantic_scores[i] if i < len(semantic_scores) else 0.0,
                bm25_score=bm25_scores[i] if i < len(bm25_scores) else 0.0,
                ngram_score=ngram_scores[i] if i < len(ngram_scores) else 0.0,
                fuzzy_score=fuzzy_scores[i] if i < len(fuzzy_scores) else 0.0
            )
            feature_dicts.append(feats)
        
        return pd.DataFrame(feature_dicts)
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        # Extract from a dummy example
        dummy_features = self.extract_single("test", "test query")
        return list(dummy_features.keys())


def normalize_features(df: pd.DataFrame, feature_cols: List[str], clip_quantile: float = 0.99):
    """
    Normalize features using robust scaling.
    Clips outliers and standardizes.
    """
    df = df.copy()
    
    for col in feature_cols:
        if col in df.columns:
            # Clip outliers
            q_high = df[col].quantile(clip_quantile)
            df[col] = df[col].clip(upper=q_high)
            
            # Standardize
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0
    
    return df


def main():
    """Test feature extraction."""
    from data_loader import DataLoader
    
    logger.info("Testing feature extraction...")
    
    # Load data
    loader = DataLoader()
    query_features_map = loader.get_query_to_features_map()
    
    # Create extractor
    extractor = FeatureExtractor(query_features_map)
    
    # Test examples
    test_cases = [
        ("ifon 16", "iphone 16", 0.85, 2.5, 0.7, 0.8),
        ("blaek blaijer", "black blazer", 0.75, 1.2, 0.6, 0.75),
        ("running", "running shoes for men", 0.9, 3.0, 0.8, 0.6),
    ]
    
    print("\nFeature Extraction Examples:")
    print("="*80)
    
    for prefix, query, sem, bm25, ng, fz in test_cases:
        features = extractor.extract_single(prefix, query, sem, bm25, ng, fz)
        
        print(f"\nPrefix: '{prefix}' -> Query: '{query}'")
        print("-"*80)
        
        # Show top features
        important_features = [
            "semantic_cosine", "bm25_score", "edit_distance_ratio",
            "exact_prefix_match", "token_jaccard", "popularity_score",
            "log_orders", "log_volume"
        ]
        
        for feat in important_features:
            if feat in features:
                print(f"  {feat:25s}: {features[feat]:.4f}")
    
    # Show all feature names
    print(f"\n\nTotal features: {len(extractor.get_feature_names())}")
    print("Feature names:")
    for i, name in enumerate(extractor.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()