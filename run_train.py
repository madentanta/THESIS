"""
Combined Supabase Fetch, Preprocess, and Train Script for Hopular Pipeline

This script combines three steps using subprocess:
1. Fetch data from Supabase table (using fetch_from_supabase.py)
2. Preprocess the data (using preprocessing.py)
3. Train the Hopular model (using trainer.py)
"""

import subprocess
import sys
import argparse
import os
from typing import Optional


def run_command(cmd: list, description: str) -> bool:
    """
    Run a subprocess command and return True if it succeeds
    """
    print(f"\n{description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False


def run_train_pipeline(supabase_url: str, 
                       supabase_key: str, 
                       table_name: str, 
                       target_column: str,
                       data: str = "supabase_data.csv",
                       output_csv: str = "processed_data.csv",
                       filters: Optional[str] = None,
                       limit: Optional[int] = None,
                       epochs: int = 100,
                       batch_size: int = 32,
                       patience: int = 10,
                       test_size: float = 0.2,
                       min_class_samples: int = 2):
    """
    Execute the full pipeline using subprocess calls
    """
    print("=" * 70)
    print("RUN_TRAIN: Supabase Fetch, Preprocess, and Train Pipeline")
    print("=" * 70)
    
    # Step 1: Fetch data from Supabase
    print("\nStep 1: Fetching data from Supabase...")
    fetch_cmd = [
        sys.executable, "fetch_from_supabase.py",
        "--url", supabase_url,
        "--key", supabase_key,
        "--table", table_name,
        "--output", data
    ]
    
    if filters:
        fetch_cmd.extend(["--filters", filters])
    if limit:
        fetch_cmd.extend(["--limit", str(limit)])
    
    fetch_success = run_command(fetch_cmd, "Fetching data from Supabase")
    
    if not fetch_success:
        print("❌ Failed to fetch data from Supabase. Exiting.")
        return False
    
    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing data...")
    preprocess_cmd = [
        sys.executable, "preprocessing.py",
        "--input", data,
        "--output", output_csv
    ]
    
    preprocess_success = run_command(preprocess_cmd, "Preprocessing data")
    
    if not preprocess_success:
        print("❌ Failed to preprocess data. Exiting.")
        return False
    
    # Step 3: Train the model
    print("\nStep 3: Training the Hopular model...")
    train_cmd = [
        sys.executable, "trainer.py",
        "--data", output_csv,
        "--target", target_column,
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--patience", str(patience),
        "--test_size", str(test_size),
        "--min_class_samples", str(min_class_samples)
    ]
    
    train_success = run_command(train_cmd, "Training the Hopular model")
    
    if not train_success:
        print("❌ Failed to train the model. Exiting.")
        return False
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Files generated:")
    print(f"  - Raw Supabase data: {data}")
    print(f"  - Processed training data: {output_csv}")
    print(f"  - Trained model: best_hopular_model.pt")
    print(f"  - Metadata: metadata.pkl")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Fetch from Supabase, preprocess, and train Hopular model"
    )
    
    # Supabase arguments
    parser.add_argument(
        "--url", 
        type=str, 
        required=True,
        help="Supabase URL (e.g., https://abcde.supabase.co)"
    )
    
    parser.add_argument(
        "--key", 
        type=str, 
        required=True,
        help="Supabase API key (anon or service role key)"
    )
    
    parser.add_argument(
        "--table", 
        type=str, 
        required=True,
        help="Name of the Supabase table to fetch"
    )
    
    parser.add_argument(
        "--target", 
        type=str, 
        required=True,
        help="Target column name for training"
    )
    
    parser.add_argument(
        "--supabase-csv", 
        type=str, 
        default="data.csv",
        help="CSV filename for raw Supabase data (default: data.csv)"
    )
    
    parser.add_argument(
        "--output-csv", 
        type=str, 
        default="processed_data.csv",
        help="Output CSV filename after preprocessing (default: processed_data.csv)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10,
        help="Early stopping patience (default: 10)"
    )
    
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Test size fraction (default: 0.2)"
    )
    
    parser.add_argument(
        "--min-class-samples", 
        type=int, 
        default=2,
        help="Minimum samples per class before filtering (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Run the complete pipeline
    success = run_train_pipeline(
        supabase_url=args.url,
        supabase_key=args.key,
        table_name=args.table,
        target_column=args.target,
        data=args.data,
        output_csv=args.output_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        test_size=args.test_size,
        min_class_samples=args.min_class_samples
    )
    
    if success:
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()