# Hardcoded Scoring System

## Overview
Removed LightGBM dependency and replaced with a fast, hardcoded scoring function that combines retrieval scores and features with tuned weights.

## Changes Made

### 1. Removed LightGBM Components
- **retrieval_engine.py**: Removed `lightgbm` import and model loading
- **requirements.txt**: Commented out `lightgbm>=4.0.0`
- **train_ranker.py**: No longer needed for inference (keep for reference only)

### 2. New Hardcoded Scoring Function

The `_compute_scores()` method in `RetrievalEngine` now uses a weighted combination of features:

```python
score = (
    exact_match * 100.0 +           # Exact match is king
    starts_with * 50.0 +            # Prefix match is very important
    sem_score * 30.0 +              # Semantic similarity
    partial_ratio * 25.0 +          # Partial fuzzy match
    bm25_score * 20.0 +             # BM25 score
    orders * 15.0 +                 # Order popularity
    ratio * 12.0 +                  # Full fuzzy ratio
    ngram_score * 10.0 +            # N-gram similarity
    volume * 8.0 +                  # Search volume
    clicks * 6.0 +                  # Catalog clicks
    contains * 5.0 +                # Contains prefix
    views * 4.0 +                   # Catalog views
    len_ratio * 3.0 +               # Length similarity
    fuzzy_score * 2.0 -             # Fuzzy retrieval score
    len_diff * 0.5                  # Penalize length difference
)
```

### 3. Features Used

**Retrieval Scores:**
- `sem_score`: Semantic similarity from sentence transformers
- `bm25_score`: BM25 lexical match score (normalized)
- `ngram_score`: Character n-gram similarity
- `fuzzy_score`: Fuzzy matching score

**String Matching:**
- `exact_match`: Binary (1.0 if exact match)
- `starts_with`: Binary (1.0 if query starts with prefix)
- `contains`: Binary (1.0 if query contains prefix)
- `ratio`: Full fuzzy ratio (RapidFuzz)
- `partial_ratio`: Partial fuzzy ratio

**Length Features:**
- `len_diff`: Absolute length difference
- `len_ratio`: Ratio of min/max lengths

**Popularity Features (log-transformed):**
- `orders`: Log(1 + orders) / 10.0
- `volume`: Log(1 + volume) / 15.0
- `clicks`: Log(1 + catalog_clicks) / 12.0
- `views`: Log(1 + catalog_views) / 12.0

## Benefits

1. **Speed**: No model training or loading required
2. **Simplicity**: Easy to understand and tune weights
3. **Transparency**: Clear feature importance visible in weights
4. **No Dependencies**: Removed LightGBM dependency
5. **Deterministic**: Same input always produces same output

## Performance

The hardcoded weights were tuned based on feature importance from previous LightGBM runs:
- Exact match and prefix matching get highest weights
- Semantic and fuzzy matching provide robustness
- Popularity features help with ranking

## Usage

```python
from retrieval_engine import RetrievalEngine

# Initialize (no model training needed!)
engine = RetrievalEngine()

# Get completions
results = engine.retrieve("running sho")
# ['running shoes', 'running shoes for men', ...]
```

## Files Modified

- `retrieval_engine.py`: Replaced LightGBM with hardcoded scoring
- `requirements.txt`: Removed LightGBM dependency
- Added `rapidfuzz` for fuzzy matching features

## Files No Longer Needed for Inference

- `train_ranker.py`: Training script (keep for reference)
- `feature_extractor.py`: Feature extraction (keep for reference)
- `models/lightgbm_ranker.txt`: Model file (not loaded anymore)
- `artifacts/train_features.pkl`: Training data (not needed)

## Future Improvements

If you want to tune the weights:
1. Edit the `_compute_scores()` method in `retrieval_engine.py`
2. Adjust the weight values based on your evaluation metrics
3. Test on a validation set
4. No retraining needed - just restart the server!
