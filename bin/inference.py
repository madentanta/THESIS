"""
Hopular Inference Module
Provides prediction capabilities for trained Hopular models
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Union, List, Dict, Optional, Tuple
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from bin.customhopular import Hopular, encode_features


class HopularInference:
    """
    Inference wrapper for Hopular model that handles preprocessing, prediction, and postprocessing
    """

    def __init__(self,
                 model_path: str = 'best_hopular_model.pt',
                 metadata_path: str = 'metadata.pkl',
                 device: str = None):
        """
        Initialize the inference module

        Args:
            model_path: Path to the trained model checkpoint
            metadata_path: Path to the metadata pickle file (contains scalers, encoders, etc.)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            raise FileNotFoundError(f"Metadata file {metadata_path} not found. Training must save metadata.pkl for inference.")

        # Load the trained model
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"Hopular inference module loaded on {self.device}")
        print(f"Task: {self.metadata.get('task', 'Unknown')}")

    def _load_model(self, model_path: str) -> Hopular:
        """Load the trained Hopular model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        # Create model with metadata configuration
        model = Hopular(
            input_sizes=self.metadata['input_sizes'],
            target_discrete=self.metadata['target_discrete'],
            target_numeric=self.metadata['target_numeric'],
            feature_discrete=self.metadata['feature_discrete'],
            memory=self.metadata['memory'],  # Use training memory
            feature_size=32,  # Use same configuration as training
            hidden_size=16,
            hidden_size_factor=1.0,
            num_heads=4,
            num_blocks=1,
            scaling_factor=1.0,
            input_dropout=0.2,
            lookup_dropout=0.2,
            output_dropout=0.2,
            memory_ratio=1.0  # Use full memory during inference
        )

        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)

        return model

    def preprocess_input(self,
                        input_data: Union[pd.DataFrame, np.ndarray, dict],
                        feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess input data to match the format expected by the model

        Args:
            input_data: Input data as DataFrame, array, or dict
            feature_columns: Names of feature columns (if different from training data)

        Returns:
            Tuple of (preprocessed_data, feature_names)
        """
        if self.metadata is None:
            raise ValueError("Metadata is required for preprocessing")

        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif isinstance(input_data, np.ndarray):
            if feature_columns is None:
                raise ValueError("feature_columns must be provided when input is numpy array")
            input_data = pd.DataFrame(input_data, columns=feature_columns)
        elif not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be DataFrame, dict, or numpy array")

        # Get feature names from metadata
        # For newer scikit-learn versions, try to use scaler's feature_names_in_
        # For older versions, use original_features stored in metadata
        scaler = self.metadata.get('scaler')
        if hasattr(scaler, 'feature_names_in_'):
            original_features = scaler.feature_names_in_
        else:
            # Fallback to original_features stored in metadata
            original_features = self.metadata.get('original_features', [])

        # Validate that we have the required feature names
        if not original_features:
            raise ValueError("Could not determine original feature names from metadata")

        print(f"Expected features: {len(original_features)}, Input features: {len(input_data.columns)}")

        # Ensure correct feature ordering
        missing_cols = set(original_features) - set(input_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        input_df = input_data[original_features].copy()

        # Apply the same preprocessing as during training
        # Encode categorical columns
        label_encoders = self.metadata.get('label_encoders', {})
        for col in label_encoders.keys():
            if col in input_df.columns:
                # Handle unseen categories by mapping them to a default value
                le = label_encoders[col]
                # Get the list of known classes
                known_classes = set(le.classes_)

                # Convert to string first to handle different data types
                input_df[col] = input_df[col].astype(str)

                # Map unknown values to the first known class to avoid unseen label errors
                input_df[col] = input_df[col].apply(
                    lambda x: x if x in known_classes else str(le.classes_[0])
                )

                # Transform using the fitted encoder
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError as e:
                    # Handle unseen labels by setting them to the first class
                    print(f"Warning: Unseen labels in {col}, mapping to first class")
                    input_df[col] = input_df[col].apply(
                        lambda x: le.transform([str(le.classes_[0])])[0] if x not in known_classes else le.transform([str(x)])[0]
                    )

        # Apply the same standardization as during training
        scaler = self.metadata.get('scaler')
        X_scaled = scaler.transform(input_df.values.astype(np.float32))

        # Encode features in the same format as training
        # We only encode the feature part (not the target part) so exclude target from input_sizes
        n_features = len(original_features)  # Number of input features (excluding target)
        input_sizes = self.metadata['input_sizes'][:n_features]  # Only feature sizes, exclude target
        feature_discrete = [i for i in self.metadata['feature_discrete'] if i < len(input_sizes)]

        print(f"Input shape before encoding: {X_scaled.shape}")
        print(f"Input sizes: {input_sizes}")
        print(f"Feature discrete: {feature_discrete}")

        X_encoded_features = encode_features(X_scaled, input_sizes, feature_discrete)

        print(f"Encoded features shape: {X_encoded_features.shape}")

        # The model expects the full input including target placeholders
        # The actual input_sizes for the model includes both features and target
        total_input_sizes = self.metadata['input_sizes']  # Includes features + target
        total_feature_size = sum(total_input_sizes)

        # Create full input with feature part + empty target part
        batch_size = X_encoded_features.shape[0]
        X_full = np.zeros((batch_size, total_feature_size), dtype=np.float32)

        # Place the encoded features at the beginning
        feature_size_sum = sum(input_sizes)
        X_full[:, :feature_size_sum] = X_encoded_features

        # Leave target part as zeros (to be predicted)

        print(f"Final input shape: {X_full.shape}, Expected: {total_feature_size}")

        return X_full, list(original_features)

    def predict(self,
               input_data: Union[pd.DataFrame, np.ndarray, dict],
               feature_columns: Optional[List[str]] = None,
               return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on input data

        Args:
            input_data: Input data for prediction
            feature_columns: Names of feature columns (if input is array)
            return_probabilities: Whether to return probability/confidence scores (classification only)

        Returns:
            Predictions as numpy array, optionally with probabilities
        """
        with torch.no_grad():
            # Preprocess input
            X_encoded, feature_names = self.preprocess_input(input_data, feature_columns)

            # Convert to tensor
            X_tensor = torch.FloatTensor(X_encoded).to(self.device)

            # For inference, we typically don't want to use memory masking as in training
            # Set memory_mask to None to allow the model to access its full memory
            memory_mask = None

            # Forward pass
            output = self.model(X_tensor, memory_mask=memory_mask)

            # Extract target predictions from the output
            # The target is always at the end based on input_sizes
            input_sizes = self.metadata['input_sizes']
            target_discrete = self.metadata['target_discrete']
            target_numeric = self.metadata['target_numeric']

            # Determine target start index (sum of all feature input sizes)
            feature_sizes_sum = sum(input_sizes[:-1])  # All feature sizes before target
            target_size = input_sizes[-1]  # Size of target
            target_start = feature_sizes_sum
            target_end = target_start + target_size

            # Extract target predictions
            target_predictions = output[:, target_start:target_end]

            # Convert back to original format
            predictions = self._postprocess_predictions(target_predictions)

            if return_probabilities and target_discrete and len(target_discrete) > 0:
                # For classification, return both predictions and probabilities
                probabilities = torch.softmax(target_predictions, dim=1).cpu().numpy()
                return predictions, probabilities
            else:
                return predictions

    def _postprocess_predictions(self, target_predictions: torch.Tensor) -> np.ndarray:
        """
        Convert model predictions back to original format

        Args:
            target_predictions: Raw model output for target

        Returns:
            Processed predictions in original format
        """
        if self.metadata is None:
            return target_predictions.cpu().numpy()

        task = self.metadata.get('task', 'classification')
        target_discrete = self.metadata.get('target_discrete', [])
        target_numeric = self.metadata.get('target_numeric', [])

        if task == 'classification' or len(target_discrete) > 0:
            # For classification, take the argmax to get predicted class
            predictions = torch.argmax(target_predictions, dim=1).cpu().numpy()

            # If we have a target label encoder, inverse transform
            target_encoder = self.metadata.get('target_label_encoder')
            if target_encoder is not None:
                predictions = target_encoder.inverse_transform(predictions)
        elif task == 'regression' or len(target_numeric) > 0:
            # For regression, the output needs to be denormalized
            # Get the target scaler if available
            predictions = target_predictions.cpu().numpy()

            # Since regression targets were standardized during training,
            # we might need to apply inverse scaling if available.
            # For now, return the raw predictions.
            # Note: In the training setup, regression targets are not one-hot encoded,
            # so the output dimension is just 1 for the target value.
        else:
            # Default case - return raw values
            predictions = target_predictions.cpu().numpy()

        return predictions

    def predict_from_file(self,
                         file_path: str,
                         target_column: str = None,
                         return_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions directly from a CSV file

        Args:
            file_path: Path to input CSV file
            target_column: Name of target column (if we want to exclude it)
            return_dataframe: Whether to return as DataFrame with predictions

        Returns:
            Predictions as numpy array or DataFrame with predictions
        """
        df = pd.read_csv(file_path)

        if target_column and target_column in df.columns:
            # Remove target column if present in input
            X_df = df.drop(columns=[target_column])
        else:
            X_df = df

        predictions = self.predict(X_df)

        if return_dataframe:
            result_df = df.copy()
            result_df['prediction'] = predictions
            return result_df
        else:
            return predictions


def load_trained_model(model_path: str = 'best_hopular_model.pt',
                      metadata_path: str = 'metadata.pkl',
                      device: str = None) -> HopularInference:
    """
    Convenience function to load a trained Hopular model for inference

    Args:
        model_path: Path to the trained model checkpoint
        metadata_path: Path to the metadata pickle file
        device: Device to run inference on

    Returns:
        HopularInference instance
    """
    return HopularInference(model_path, metadata_path, device)


def predict_new_data(input_data: Union[pd.DataFrame, np.ndarray, dict],
                    model_path: str = 'best_hopular_model.pt',
                    metadata_path: str = 'metadata.pkl',
                    feature_columns: Optional[List[str]] = None,
                    device: str = None) -> np.ndarray:
    """
    Convenience function to make predictions with a trained model

    Args:
        input_data: Input data for prediction
        model_path: Path to the trained model checkpoint
        metadata_path: Path to the metadata pickle file
        feature_columns: Names of feature columns (if input is array)
        device: Device to run inference on

    Returns:
        Predictions as numpy array
    """
    inference_model = load_trained_model(model_path, metadata_path, device)
    return inference_model.predict(input_data, feature_columns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hopular Model Inference")
    parser.add_argument("--model_path", type=str, default="best_hopular_model.pt",
                       help="Path to the trained model checkpoint")
    parser.add_argument("--metadata_path", type=str, default="metadata.pkl",
                       help="Path to the metadata file")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to input CSV file for prediction")
    parser.add_argument("--target_column", type=str, default=None,
                       help="Name of target column (to exclude from prediction)")
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                       help="Path to save predictions")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run inference on (cuda/cpu) - defaults to auto")

    args = parser.parse_args()

    print("Loading Hopular model for inference...")
    hopular_inference = HopularInference(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        device=args.device
    )

    print("Making predictions...")
    predictions_df = hopular_inference.predict_from_file(
        file_path=args.input_file,
        target_column=args.target_column,
        return_dataframe=True
    )

    print(f"Saving predictions to {args.output_file}")
    predictions_df.to_csv(args.output_file, index=False)

    print("Predictions completed!")
    print(f"Shape of predictions: {predictions_df.shape}")
    print("First 10 predictions:")
    print(predictions_df['prediction'].head(10))