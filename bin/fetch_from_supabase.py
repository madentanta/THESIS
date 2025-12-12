"""
Supabase to CSV Fetcher for Hopular Pipeline

This script connects to a Supabase database and downloads a table as a CSV file
to use in the Hopular pipeline.
"""

import os
import pandas as pd
from typing import Optional
import argparse


def fetch_table_from_supabase(
    supabase_url: str,
    supabase_key: str,
    table_name: str,
    output_file: str = "data.csv",
    query_filters: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    Fetch a table from Supabase and save as CSV
    
    Args:
        supabase_url: Your Supabase project URL
        supabase_key: Your Supabase anon/api key
        table_name: Name of the table to fetch
        output_file: Output CSV filename (default: data.csv)
        query_filters: Optional filters in query format (e.g., "status.eq.active")
        limit: Optional row limit
    """
    try:
        # Import supabase client - install via: pip install supabase
        from supabase import create_client, Client
        
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        print(f"Connecting to Supabase: {supabase_url}")
        print(f"Fetching table: {table_name}")
        
        # Build the query
        query = supabase.table(table_name).select("*")
        
        # Add filters if provided
        if query_filters:
            # Parse filters and apply them
            for filter_str in query_filters.split(","):
                filter_str = filter_str.strip()
                if ".eq." in filter_str:
                    col, val = filter_str.split(".eq.")
                    query = query.eq(col.strip(), val.strip())
                elif ".gt." in filter_str:
                    col, val = filter_str.split(".gt.")
                    query = query.gt(col.strip(), val.strip())
                elif ".lt." in filter_str:
                    col, val = filter_str.split(".lt.")
                    query = query.lt(col.strip(), val.strip())
                elif ".gte." in filter_str:
                    col, val = filter_str.split(".gte.")
                    query = query.gte(col.strip(), val.strip())
                elif ".lte." in filter_str:
                    col, val = filter_str.split(".lte.")
                    query = query.lte(col.strip(), val.strip())
                elif ".like." in filter_str:
                    col, val = filter_str.split(".like.")
                    query = query.like(col.strip(), val.strip())
        
        # Add limit if provided
        if limit:
            query = query.limit(limit)
        
        # Execute the query
        response = query.execute()
        data = response.data
        
        if not data:
            print(f"Warning: No data found in table '{table_name}'")
            return
            
        print(f"Retrieved {len(data)} rows from '{table_name}' table")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        
        # Print basic info about the dataset
        print(f"\nDataset Info:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names: {list(df.columns)}")
        
        return df
        
    except ImportError:
        print("Error: Supabase client not found. Install it with: pip install supabase")
        return None
    except Exception as e:
        print(f"Error fetching data from Supabase: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch table from Supabase and save as CSV for Hopular pipeline")
    
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
        help="Name of the table to fetch"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="data.csv",
        help="Output CSV filename (default: data.csv)"
    )
    
    args = parser.parse_args()
    
    # Fetch data from Supabase
    df = fetch_table_from_supabase(
        supabase_url=args.url,
        supabase_key=args.key,
        table_name=args.table,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()