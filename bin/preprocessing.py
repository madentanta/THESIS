import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    # ---------------------------------------
    # Argument parser
    # ---------------------------------------
    parser = argparse.ArgumentParser(
        description="Preprocess the PlantAdvisor dataset"
    )

    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output", type=str, default="processed data.csv",
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    # ---------------------------------------
    # Load CSV
    # ---------------------------------------
    print(f"📥 Loading dataset: {args.input}")
    df = pd.read_csv(args.input, header=0)

    print("✅ Columns detected:", df.columns.tolist())

    # ---------------------------------------
    # Fill missing coordinates
    # ---------------------------------------
    if "Longitude" in df.columns and "Latitude" in df.columns:
        df["Longitude"] = df["Longitude"].fillna(method="ffill").fillna(method="bfill")
        df["Latitude"] = df["Latitude"].fillna(method="ffill").fillna(method="bfill")
        print("📌 Latitude & Longitude filled")
    else:
        print("⚠️ WARNING: No Longitude/Latitude columns found")

    # ---------------------------------------
    # Drop unused columns if present
    # ---------------------------------------
    for col in ["Nama Daerah", "Evidence"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"🗑 Removed column: {col}")

    # ---------------------------------------
    # Encode categorical columns
    # ---------------------------------------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print("📂 Categorical columns detected:", cat_cols)

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"🔠 Encoded: {col}")

    # ---------------------------------------
    # Save to CSV
    # ---------------------------------------
    df.to_csv(args.output, index=False)
    print(f"💾 Saved processed dataset → {args.output}")


if __name__ == "__main__":
    main()
