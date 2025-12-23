import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def compute_feature_bounds(df, continuous_cols, q_low=0.01, q_high=0.99):
    """
    Compute robust bounds for augmentation validation
    """
    bounds = {}
    for col in continuous_cols:
        bounds[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].quantile(q_low),
            "max": df[col].quantile(q_high),
        }
    return bounds

def augment_gaussian_noise_validated(
    df: pd.DataFrame,
    continuous_cols: list,
    bounds: dict,
    n_copies: int = 10,
    sigma: float = 0.03,
    std_clip: float = 3.0,
):
    """
    Gaussian augmentation WITH validation:
    - clipped by quantiles
    - clipped by std range
    """
    augmented = []

    for _ in range(n_copies):
        noisy = df.copy()

        noise = np.random.normal(
            loc=0.0,
            scale=sigma,
            size=noisy[continuous_cols].shape,
        )

        noisy[continuous_cols] += noise

        # ---- VALIDATION ----
        for col in continuous_cols:
            b = bounds[col]

            # std-based clipping
            noisy[col] = noisy[col].clip(
                lower=b["mean"] - std_clip * b["std"],
                upper=b["mean"] + std_clip * b["std"],
            )

            # quantile-based clipping (robust)
            noisy[col] = noisy[col].clip(
                lower=b["min"],
                upper=b["max"],
            )

        augmented.append(noisy)

    df_aug = pd.concat([df] + augmented, ignore_index=True)
    return df_aug

def apply_physical_rules(df, bounds):
    """
    Validate augmented samples using training-data quantiles
    """
    mask = np.ones(len(df), dtype=bool)

    for col, b in bounds.items():
        mask &= (df[col] >= b["min"]) & (df[col] <= b["max"])

    return df[mask]



def main():
    # ---------------------------------------
    # Argument parser
    # ---------------------------------------
    parser = argparse.ArgumentParser(
        description="Preprocess PlantAdvisor / Hopular dataset (augmentation is mandatory)"
    )

    parser.add_argument(
        "--input", type=str, default="data/data.csv",
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output", type=str, default="data/processed_data.csv",
        help="Path to output CSV file"
    )

    parser.add_argument(
        "--augment_copies", type=int, default=25,
        help="Number of augmented copies PER sample"
    )

    parser.add_argument(
        "--augment_sigma", type=float, default=0.03,
        help="Gaussian noise standard deviation (recommended: 0.02–0.05)"
    )

    args = parser.parse_args()

    # ---------------------------------------
    # Load CSV
    # ---------------------------------------
    print(f"Loading dataset: {args.input}")
    df = pd.read_csv(args.input, header=0)
    print("Columns detected:", df.columns.tolist())

    # ---------------------------------------
    # Drop unused columns
    # ---------------------------------------
    cols_to_drop = ["id", "nama_daerah", "created_at"]

    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Removed column: {col}")

    # ---------------------------------------
    # Encode categorical columns
    # ---------------------------------------
    categorical_columns = ["kecamatan", "nama_tanaman"]
    encoders = {}

    for col in categorical_columns:
        if col in df.columns:
            print(f"Encoding column: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"Classes in {col}: {le.classes_}")

    # ---------------------------------------
    # Normalize continuous features
    # ---------------------------------------
    continuous_cols = [
        "fertility",
        "moisture",
        "ph",
        "temp",
        "sunlight",
        "humidity",
    ]

    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    # ---------------------------------------
    # MANDATORY AUGMENTATION
    # ---------------------------------------
    print("\n=== Mandatory Data Augmentation ===")
    print(f"Original samples: {len(df)}")
    print(f"Copies per sample: {args.augment_copies}")
    print(f"Gaussian sigma: {args.augment_sigma}")

    bounds = compute_feature_bounds(
        df, 
        continuous_cols,
        q_low=0.01,
        q_high=0.99
    )


    df = augment_gaussian_noise_validated(
        df=df,
        continuous_cols=continuous_cols,
        bounds=bounds,
        n_copies=args.augment_copies,
        sigma=args.augment_sigma,
        std_clip=3.0,
    )


    print(f"Augmented samples: {len(df)}")

    df = apply_physical_rules(df, bounds)

    # ---------------------------------------
    # Target distribution check
    # ---------------------------------------
    target_col = "nama_tanaman"
    if target_col in df.columns:
        print("\nTarget distribution after augmentation:")
        print(df[target_col].value_counts())
        print(f"Number of classes: {df[target_col].nunique()}")

    # ---------------------------------------
    # Save to CSV
    # ---------------------------------------
    df.to_csv(args.output, index=False)
    print(f"\nSaved processed dataset -> {args.output}")

    preprocess_metadata = {
        "feature_names": [
            "fertility",
            "moisture",
            "ph",
            "temp",
            "sunlight",
            "humidity",
            "kecamatan",
        ],
        "continuous_cols": [
            "fertility",
            "moisture",
            "ph",
            "temp",
            "sunlight",
            "humidity",
        ],
        "categorical_cols": ["kecamatan"],
        "target_name": "nama_tanaman",
        "label_encoders": encoders,
        "scaler": scaler,
    }
    import pickle

    with open("data/preprocess_metadata.pkl", "wb") as f:
        pickle.dump(preprocess_metadata, f)

    print("Saved preprocessing metadata -> data/preprocess_metadata.pkl")




if __name__ == "__main__":
    main()
