"""
Generate predictions for test set.
Creates submission file with top-10 completions for each prefix.
"""
import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
import time

from config import *
from data_loader import DataLoader
from retrieval_engine import RetrievalEngine

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def generate_submission(output_path: str = "submission.csv"):
    """Generate submission file for test prefixes."""
    
    logger.info("="*80)
    logger.info("GENERATING SUBMISSION FILE")
    logger.info("="*80)
    
    # Load test prefixes
    loader = DataLoader()
    test_df = loader.load_test_prefixes()
    
    logger.info(f"Loaded {len(test_df):,} test prefixes")
    
    # Initialize retrieval engine
    logger.info("Initializing retrieval engine...")
    engine = RetrievalEngine(use_cache=True)
    
    # Generate predictions
    logger.info("Generating predictions...")
    results = []
    
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        prefix = row["prefix"]
        
        # Get top-10 completions
        try:
            completions = engine.retrieve(prefix)
            
            # Ensure we have exactly 10 results (pad if necessary)
            while len(completions) < FINAL_TOP_K:
                completions.append("")  # Empty string for missing
            
            completions = completions[:FINAL_TOP_K]
            
            # Add to results
            result_row = {"prefix": prefix}
            for i, completion in enumerate(completions, 1):
                result_row[f"completion_{i}"] = completion
            
            results.append(result_row)
            
        except Exception as e:
            logger.error(f"Error processing prefix '{prefix}': {e}")
            # Add empty row
            result_row = {"prefix": prefix}
            for i in range(1, FINAL_TOP_K + 1):
                result_row[f"completion_{i}"] = ""
            results.append(result_row)
    
    elapsed = time.time() - start_time
    
    # Create submission dataframe
    submission_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = Path(output_path)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Submission file saved to: {output_path}")
    logger.info(f"Total prefixes: {len(submission_df):,}")
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"Average time per prefix: {elapsed/len(submission_df)*1000:.2f}ms")
    logger.info(f"{'='*80}\n")
    
    # Show sample
    logger.info("Sample predictions:")
    print(submission_df.head(10).to_string(index=False))
    
    return submission_df


def evaluate_on_train_sample(sample_size: int = 1000):
    """
    Evaluate Hit@10 on a sample of training data.
    Useful for quick validation.
    """
    logger.info("="*80)
    logger.info("EVALUATION ON TRAINING SAMPLE")
    logger.info("="*80)
    
    # Load training data
    loader = DataLoader()
    train_df = loader.load_train_data()
    
    # Sample
    if sample_size and sample_size < len(train_df):
        eval_df = train_df.sample(n=sample_size, random_state=42)
    else:
        eval_df = train_df
    
    logger.info(f"Evaluating on {len(eval_df):,} examples")
    
    # Initialize engine
    engine = RetrievalEngine(use_cache=False)
    
    # Evaluate
    hits = 0
    total = 0
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
        prefix = row["prefix"]
        ground_truth = row["query"]
        
        try:
            predictions = engine.retrieve(prefix)
            
            # Check if ground truth is in predictions
            if ground_truth in predictions:
                hits += 1
            
            total += 1
            
        except Exception as e:
            logger.error(f"Error evaluating prefix '{prefix}': {e}")
            total += 1
    
    hit_rate = hits / total if total > 0 else 0
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total examples: {total:,}")
    logger.info(f"Hits: {hits:,}")
    logger.info(f"Hit@10: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    logger.info(f"{'='*80}\n")
    
    return hit_rate


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate submission or evaluate")
    parser.add_argument(
        "--mode",
        choices=["submission", "evaluate"],
        default="submission",
        help="Mode: 'submission' to generate test predictions, 'evaluate' to check training accuracy"
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output file path for submission"
    )
    parser.add_argument(
        "--eval_sample",
        type=int,
        default=1000,
        help="Number of training examples to evaluate on"
    )
    
    args = parser.parse_args()
    
    if args.mode == "submission":
        generate_submission(args.output)
    elif args.mode == "evaluate":
        evaluate_on_train_sample(args.eval_sample)


if __name__ == "__main__":
    main()