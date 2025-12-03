"""
Data loading utilities for autocomplete system.
Loads and caches data from HuggingFace Hub.
"""
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import pickle
from typing import Tuple, Dict
import logging

from config import *

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles all data loading with caching."""
    
    def __init__(self, force_reload: bool = False):
        self.force_reload = force_reload
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def _get_cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.parquet"
    
    def _load_with_cache(self, name: str, data_files: str) -> pd.DataFrame:
        """Load dataset with caching."""
        cache_path = self._get_cache_path(name)
        
        if not self.force_reload and cache_path.exists():
            logger.info(f"Loading {name} from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        
        logger.info(f"Downloading {name} from HuggingFace...")
        dataset = load_dataset(
            HF_DATASET_NAME,
            data_files=data_files,
            split="train"
        )
        df = dataset.to_pandas()
        
        # Cache for future use
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached {name} to {cache_path}")
        
        return df
    
    def load_train_data(self) -> pd.DataFrame:
        """Load training data (prefix -> query pairs)."""
        df = self._load_with_cache("train", HF_TRAIN_FILES)
        logger.info(f"Loaded {len(df):,} training examples")
        return df
    
    def load_query_pool(self) -> pd.DataFrame:
        """Load candidate query pool."""
        df = self._load_with_cache("pool", HF_POOL_FILES)
        logger.info(f"Loaded {len(df):,} queries in pool")
        return df
    
    def load_query_features(self) -> pd.DataFrame:
        """Load query engagement features."""
        df = self._load_with_cache("features", HF_FEATURES_FILES)
        
        # Fill NaN with 0 for numeric columns
        for col in POPULARITY_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        logger.info(f"Loaded features for {len(df):,} queries")
        return df
    
    def load_test_prefixes(self) -> pd.DataFrame:
        """Load test prefixes."""
        df = self._load_with_cache("test", HF_TEST_FILES)
        logger.info(f"Loaded {len(df):,} test prefixes")
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets at once."""
        return (
            self.load_train_data(),
            self.load_query_pool(),
            self.load_query_features(),
            self.load_test_prefixes()
        )
    
    def get_query_to_features_map(self) -> Dict[str, Dict]:
        """Get dictionary mapping query -> features for fast lookup."""
        cache_path = self.cache_dir / "query_features_map.pkl"
        
        if not self.force_reload and cache_path.exists():
            logger.info(f"Loading query features map from cache")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        df = self.load_query_features()
        
        # Create mapping
        feature_map = {}
        for _, row in df.iterrows():
            query = row["query"]
            features = {col: row[col] for col in POPULARITY_FEATURES if col in df.columns}
            feature_map[query] = features
        
        # Cache
        with open(cache_path, "wb") as f:
            pickle.dump(feature_map, f)
        
        logger.info(f"Created feature map for {len(feature_map):,} queries")
        return feature_map


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    if LOWERCASE:
        text = text.lower()
    
    if REMOVE_SPECIAL_CHARS:
        # Keep alphanumeric and spaces
        import re
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    train_df, pool_df, features_df, test_df = loader.load_all()
    
    print(f"\nData Summary:")
    print(f"  Training pairs: {len(train_df):,}")
    print(f"  Query pool size: {len(pool_df):,}")
    print(f"  Queries with features: {len(features_df):,}")
    print(f"  Test prefixes: {len(test_df):,}")
    
    print(f"\nSample training data:")
    print(train_df.head())
    
    print(f"\nSample pool queries:")
    print(pool_df.head())
    
    print(f"\nSample features:")
    print(features_df.head())