"""
Supabase to CSV Fetcher for Hopular Pipeline
Fetches a single fixed table from Supabase and saves it as CSV.
"""

import os
import pandas as pd
import argparse
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 🔒 Static Supabase Table Name (CHANGE THIS)
TABLE_NAME = "dataset_tanaman"


if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env")


def fetch_table_from_supabase(output_file: str = "data/data.csv"):
    """
    Fetch the fixed Supabase table and save as CSV.
    
    Args:
        output_file: Path to save CSV
    """

    try:
        from supabase import create_client, Client

        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

        print(f"Connecting to Supabase: {SUPABASE_URL}")
        print(f"Fetching table: {TABLE_NAME}")

        # Fetch full table
        response = supabase.table(TABLE_NAME).select("*").execute()
        data = response.data

        if not data:
            print(f"⚠ No data found in table '{TABLE_NAME}'")
            return None

        df = pd.DataFrame(data)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

        print("\nDataset Info:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"Error fetching data from Supabase: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch static Supabase table → CSV for Hopular pipeline"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/data.csv",
        help="Output CSV path (default: data/data.csv)"
    )

    args = parser.parse_args()

    fetch_table_from_supabase(output_file=args.output)


if __name__ == "__main__":
    main()
