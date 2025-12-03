"""
Train LightGBM ranking model with hard negatives.
"""
import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from config import *
from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from lexical_indexer import LexicalIndexer
from feature_extractor import FeatureExtractor, normalize_features

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class RankerTrainer:
    """Handles training data generation and model training."""
    
    def __init__(self):
        self.loader = DataLoader()
        self.embedder = EmbeddingGenerator()
        self.lexical = LexicalIndexer()
        
        # Load all resources
        logger.info("Loading resources...")
        self.embeddings, self.metadata, self.faiss_index = self.embedder.load_embeddings_and_index()
        self.lexical.load_indices()
        
        # Get query features
        self.query_features_map = self.loader.get_query_to_features_map()
        self.feature_extractor = FeatureExtractor(self.query_features_map)
        
        # Create query to index mapping
        self.query_to_idx = {q: i for i, q in enumerate(self.metadata["queries"])}
        
    def generate_candidates(
        self,
        prefix: str,
        ground_truth: str = None
    ) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
        """
        Generate candidate queries for a prefix using multiple retrieval methods.
        
        Returns:
            queries, semantic_scores, bm25_scores, ngram_scores, fuzzy_scores
        """
        candidates = {}  # query -> (sem_score, bm25_score, ngram_score, fuzzy_score)
        
        # 1. Semantic retrieval
        prefix_emb = self.embedder.encode_queries([prefix])
        sem_distances, sem_indices = self.embedder.search(prefix_emb, k=SEMANTIC_TOP_K)
        
        for dist, idx in zip(sem_distances[0], sem_indices[0]):
            query = self.metadata["queries"][idx]
            if query not in candidates:
                candidates[query] = [0, 0, 0, 0]
            candidates[query][0] = float(dist)  # cosine similarity (already normalized)
        
        # 2. BM25 retrieval
        bm25_indices, bm25_scores = self.lexical.search_bm25(prefix, k=BM25_TOP_K)
        for idx, score in zip(bm25_indices, bm25_scores):
            query = self.lexical.queries[idx]
            if query not in candidates:
                candidates[query] = [0, 0, 0, 0]
            candidates[query][1] = float(score)
        
        # 3. N-gram retrieval
        ngram_indices, ngram_scores = self.lexical.search_ngram(prefix, k=NGRAM_TOP_K)
        for idx, score in zip(ngram_indices, ngram_scores):
            query = self.lexical.queries[idx]
            if query not in candidates:
                candidates[query] = [0, 0, 0, 0]
            candidates[query][2] = float(score)
        
        # 4. Fuzzy retrieval (only for longer prefixes)
        if len(prefix) >= MIN_PREFIX_LENGTH_FOR_FUZZY:
            fuzzy_indices, fuzzy_scores = self.lexical.search_fuzzy(prefix, k=FUZZY_TOP_K)
            for idx, score in zip(fuzzy_indices, fuzzy_scores):
                query = self.lexical.queries[idx]
                if query not in candidates:
                    candidates[query] = [0, 0, 0, 0]
                candidates[query][3] = float(score)
        
        # Ensure ground truth is included
        if ground_truth and ground_truth not in candidates:
            # Add ground truth with zero scores
            candidates[ground_truth] = [0, 0, 0, 0]
        
        # Convert to lists
        queries = list(candidates.keys())
        scores = list(candidates.values())
        
        semantic_scores = [s[0] for s in scores]
        bm25_scores = [s[1] for s in scores]
        ngram_scores = [s[2] for s in scores]
        fuzzy_scores = [s[3] for s in scores]
        
        return queries, semantic_scores, bm25_scores, ngram_scores, fuzzy_scores
    
    def generate_training_data(
        self,
        train_df: pd.DataFrame,
        sample_size: int = None,
        num_negatives: int = TOTAL_NEGATIVES_PER_POSITIVE,
        batch_size: int = 1000
    ):
        """Generate training data with hard negatives - OPTIMIZED VERSION"""
        
        if sample_size:
            train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        
        logger.info(f"Generating training data for {len(train_df):,} examples...")
        
        # ===== STEP 1: PRE-COMPUTE ALL EMBEDDINGS =====
        logger.info("STEP 1/4: Pre-computing all prefix embeddings...")
        all_prefixes = train_df["prefix"].tolist()
        all_prefix_embs = self.embedder.encode_queries(all_prefixes)
        logger.info(f"✓ Encoded {len(all_prefixes):,} prefixes")
        
        # ===== STEP 2: BATCH CANDIDATE RETRIEVAL =====
        logger.info("STEP 2/4: Retrieving candidates in batches...")
        all_candidates_data = []
        
        for batch_start in tqdm(range(0, len(train_df), batch_size), desc="Candidate retrieval"):
            batch_end = min(batch_start + batch_size, len(train_df))
            batch_df = train_df.iloc[batch_start:batch_end]
            batch_embs = all_prefix_embs[batch_start:batch_end]
            
            # Batch semantic search
            sem_distances, sem_indices = self.embedder.search(batch_embs, k=SEMANTIC_TOP_K)
            
            for i, (_, row) in enumerate(batch_df.iterrows()):
                prefix = row["prefix"]
                ground_truth = row["query"]
                candidates = {}
                
                # Semantic results
                for dist, sem_idx in zip(sem_distances[i], sem_indices[i]):
                    query = self.metadata["queries"][sem_idx]
                    if query not in candidates:
                        candidates[query] = [0.0, 0.0, 0.0, 0.0]
                    candidates[query][0] = float(dist)
                
                # BM25
                bm25_idx, bm25_sc = self.lexical.search_bm25(prefix, k=BM25_TOP_K)
                for idx, score in zip(bm25_idx, bm25_sc):
                    query = self.lexical.queries[idx]
                    if query not in candidates:
                        candidates[query] = [0.0, 0.0, 0.0, 0.0]
                    candidates[query][1] = float(score)
                
                # N-gram
                ngram_idx, ngram_sc = self.lexical.search_ngram(prefix, k=NGRAM_TOP_K)
                for idx, score in zip(ngram_idx, ngram_sc):
                    query = self.lexical.queries[idx]
                    if query not in candidates:
                        candidates[query] = [0.0, 0.0, 0.0, 0.0]
                    candidates[query][2] = float(score)
                
                # Skip fuzzy for speed (or add if needed)
                # if len(prefix) >= MIN_PREFIX_LENGTH_FOR_FUZZY:
                #     fuzzy_idx, fuzzy_sc = self.lexical.search_fuzzy(prefix, k=FUZZY_TOP_K)
                #     for idx, score in zip(fuzzy_idx, fuzzy_sc):
                #         query = self.lexical.queries[idx]
                #         if query not in candidates:
                #             candidates[query] = [0.0, 0.0, 0.0, 0.0]
                #         candidates[query][3] = float(score)
                
                # Ensure ground truth
                if ground_truth not in candidates:
                    candidates[ground_truth] = [0.0, 0.0, 0.0, 0.0]
                
                all_candidates_data.append({
                    'prefix': prefix,
                    'ground_truth': ground_truth,
                    'candidates': candidates
                })
        
        logger.info(f"✓ Retrieved candidates for {len(all_candidates_data):,} prefixes")
        
        # ===== STEP 3: PREPARE FEATURE EXTRACTION BATCHES =====
        logger.info("STEP 3/4: Preparing feature extraction batches...")
        
        batch_prefixes = []
        batch_queries = []
        batch_sem_scores = []
        batch_bm25_scores = []
        batch_ngram_scores = []
        batch_fuzzy_scores = []
        batch_labels = []
        batch_group_sizes = []
        
        for data in tqdm(all_candidates_data, desc="Sampling negatives"):
            prefix = data['prefix']
            ground_truth = data['ground_truth']
            candidates = data['candidates']
            
            queries = list(candidates.keys())
            scores = list(candidates.values())
            
            # Find positive and negatives
            positive_idx = None
            negative_indices = []
            
            for j, q in enumerate(queries):
                if q == ground_truth:
                    positive_idx = j
                else:
                    negative_indices.append(j)
            
            if positive_idx is None:
                continue
            
            # Sample negatives
            if len(negative_indices) > num_negatives:
                negative_indices = np.random.choice(
                    negative_indices,
                    size=num_negatives,
                    replace=False
                ).tolist()
            
            selected_indices = [positive_idx] + negative_indices
            
            # Collect batch data
            for idx in selected_indices:
                batch_prefixes.append(prefix)
                batch_queries.append(queries[idx])
                batch_sem_scores.append(scores[idx][0])
                batch_bm25_scores.append(scores[idx][1])
                batch_ngram_scores.append(scores[idx][2])
                batch_fuzzy_scores.append(scores[idx][3])
            
            # Labels
            query_feats = self.query_features_map.get(ground_truth, {})
            if query_feats.get("orders", 0) > 0:
                pos_label = LABEL_EXACT_MATCH
            elif query_feats.get("catalog_clicks", 0) > 0:
                pos_label = LABEL_HIGH_ENGAGEMENT
            elif query_feats.get("catalog_views", 0) > 0:
                pos_label = LABEL_MEDIUM_ENGAGEMENT
            else:
                pos_label = LABEL_EXACT_MATCH
            
            batch_labels.extend([pos_label] + [LABEL_NEGATIVE] * len(negative_indices))
            batch_group_sizes.append(len(selected_indices))
        
        logger.info(f"✓ Prepared {len(batch_prefixes):,} feature extraction examples")
        
        # ===== STEP 4: EXTRACT FEATURES IN ONE BATCH =====
        logger.info("STEP 4/4: Extracting features (THIS MAY BE SLOW)...")
        import time
        start = time.time()
        
        features_df = self.feature_extractor.extract_batch(
            batch_prefixes,
            batch_queries,
            batch_sem_scores,
            batch_bm25_scores,
            batch_ngram_scores,
            batch_fuzzy_scores
        )
        
        elapsed = time.time() - start
        logger.info(f"✓ Feature extraction took {elapsed:.1f}s ({len(batch_prefixes)/elapsed:.0f} examples/sec)")
        
        y = np.array(batch_labels)
        groups = np.array(batch_group_sizes)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training data generated:")
        logger.info(f"  Total examples: {len(features_df):,}")
        logger.info(f"  Groups (prefixes): {len(groups):,}")
        logger.info(f"  Positive: {(y > 0).sum():,}")
        logger.info(f"  Negative: {(y == 0).sum():,}")
        logger.info(f"  Features: {len(features_df.columns)}")
        logger.info(f"{'='*80}\n")
        
        return features_df, y, groups
    
    def train_model(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
        """Train LightGBM ranker."""
        
        # Split into train/val
        logger.info("Splitting into train/validation...")
        
        # Split by groups (prefixes)
        n_groups = len(groups)
        train_size = int(n_groups * (1 - VALIDATION_SPLIT))
        
        train_groups = groups[:train_size]
        val_groups = groups[train_size:]
        
        train_size = train_groups.sum()
        val_size = val_groups.sum()
        
        X_train = X.iloc[:train_size]
        y_train = y[:train_size]
        
        X_val = X.iloc[train_size:]
        y_val = y[train_size:]
        
        logger.info(f"Train: {len(X_train):,} examples in {len(train_groups):,} groups")
        logger.info(f"Val: {len(X_val):,} examples in {len(val_groups):,} groups")
        
        # Normalize features
        feature_cols = X_train.columns.tolist()
        
        if NORMALIZE_FEATURES:
            logger.info("Normalizing features...")
            X_train = normalize_features(X_train, feature_cols)
            X_val = normalize_features(X_val, feature_cols)
        
        # Train model
        logger.info("Training LightGBM ranker...")
        
        ranker = lgb.LGBMRanker(**RANKER_PARAMS)
        
        ranker.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            eval_metric="ndcg",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(period=10)
            ]
        )
        
        logger.info("Training complete!")
        
        # Feature importance
        importance = ranker.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importance
        }).sort_values("importance", ascending=False)
        
        logger.info("\nTop 15 features:")
        print(feature_importance_df.head(15).to_string(index=False))
        
        return ranker, feature_cols
    
    def save_model(self, ranker, feature_cols: List[str]):
        """Save trained model and metadata."""
        logger.info(f"Saving model to {RANKER_MODEL_PATH}")
        
        # Save LightGBM model
        ranker.booster_.save_model(str(RANKER_MODEL_PATH))
        
        # Save feature metadata
        metadata = {
            "feature_cols": feature_cols,
            "params": RANKER_PARAMS
        }
        
        metadata_path = RANKER_MODEL_PATH.with_suffix(".pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        logger.info("Model saved successfully!")


def main():
    """Main training pipeline."""
    logger.info("="*80)
    logger.info("LIGHTGBM RANKER TRAINING PIPELINE")
    logger.info("="*80)
    
    # Initialize trainer
    trainer = RankerTrainer()
    
    # Load training data
    train_df = trainer.loader.load_train_data()
    
    # Sample for faster experimentation (remove for full training)
    SAMPLE_SIZE = 50000  # Use None for full dataset
    
    # Generate training data
    X, y, groups = trainer.generate_training_data(train_df, sample_size=SAMPLE_SIZE)
    
    # Save training features for future use
    logger.info(f"Saving training features to {TRAIN_FEATURES_PATH}")
    with open(TRAIN_FEATURES_PATH, "wb") as f:
        pickle.dump({"X": X, "y": y, "groups": groups}, f)
    
    # Train model
    ranker, feature_cols = trainer.train_model(X, y, groups)
    
    # Save model
    trainer.save_model(ranker, feature_cols)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()