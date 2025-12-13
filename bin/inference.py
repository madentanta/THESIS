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
import json
warnings.filterwarnings('ignore')
from .customhopular import Hopular, encode_features


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
        with open("data/preprocess_metadata.pkl", "rb") as f:
            self.preprocess_metadata = pickle.load(f)

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

        # Define crop names mapping based on training data
        # This maps encoded class values back to crop names
        self.crop_mapping = {0: "Jagung", 1: "Padi", 2: "Kedelai"}  # Adjust these based on your actual crop names

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
            hidden_size=32,  # Increased for better performance
            hidden_size_factor=1.0,
            num_heads=4,
            num_blocks=2,  # Increased from 1 for better performance
            scaling_factor=1.0,
            input_dropout=0.1,
            lookup_dropout=0.1,
            output_dropout=0.1,
            memory_ratio=1.0  # Use full memory during inference
        )

        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)

        return model

    def preprocess_input(self, input_data):
    # -----------------------------
    # Convert input to DataFrame
    # -----------------------------
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([self._map_input_format(input_data)])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(
                [self._map_input_format(x) for x in input_data]
            )
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Unsupported input type")

        meta = self.preprocess_metadata
        feature_names = meta["feature_names"]
        continuous_cols = meta["continuous_cols"]
        categorical_cols = meta["categorical_cols"]

        # -----------------------------
        # Ensure feature order
        # -----------------------------
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0

        input_df = input_df[feature_names]

        # -----------------------------
        # Encode categorical columns
        # -----------------------------
        for col in categorical_cols:
            le = meta["label_encoders"][col]
            known = set(le.classes_)
            default = le.classes_[0]

            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].apply(
                lambda x: x if x in known else default
            )
            input_df[col] = le.transform(input_df[col])

        # -----------------------------
        # Scale continuous features
        # -----------------------------
        scaler = meta["scaler"]
        input_df[continuous_cols] = scaler.transform(
            input_df[continuous_cols].astype(np.float32)
        )

        # -----------------------------
        # Encode for Hopular
        # -----------------------------
        X = input_df.values.astype(np.float32)

        n_features = len(feature_names)
        input_sizes = self.metadata["input_sizes"][:n_features]
        feature_discrete = [
            i for i in self.metadata["feature_discrete"]
            if i < n_features
        ]

        X_encoded = encode_features(X, input_sizes, feature_discrete)

        # -----------------------------
        # Add target placeholder
        # -----------------------------
        total_size = sum(self.metadata["input_sizes"])
        X_full = np.zeros((X_encoded.shape[0], total_size), dtype=np.float32)
        X_full[:, :sum(input_sizes)] = X_encoded

        return X_full, feature_names


    def _map_input_format(self, input_dict: dict) -> dict:
        """
        Map the API input format to the training format.
        Input: {
            "soil_ph": float,
            "temperature": float,
            "humidity": float,
            "location": string,
            "previous_crop": string
        }
        Output: {
            "fertility": float,
            "moisture": float,
            "ph": float,
            "temp": float,
            "sunlight": float,
            "humidity": float,
            "kecamatan": string,
            "nama_tanaman": string (will be ignored for prediction)
        }
        """
        # Create a mapping from API format to training format
        # Note: Some fields like soil_ph and pH need to be mapped appropriately
        mapped = {}

        # Map values from input format to training format
        # Since we don't have complete mapping in the original training data,
        # we'll use placeholder values for fields not provided in the API input
        # The most important thing is to maintain the correct column names and order

        # Map the available fields
        if 'soil_ph' in input_dict:
            mapped['ph'] = float(input_dict['soil_ph'])
        elif 'ph' in input_dict:
            mapped['ph'] = float(input_dict['ph'])

        if 'temperature' in input_dict:
            mapped['temp'] = float(input_dict['temperature'])
        elif 'temp' in input_dict:
            mapped['temp'] = float(input_dict['temp'])

        if 'humidity' in input_dict:
            mapped['humidity'] = float(input_dict['humidity'])

        # Map location to kecamatan (location field in training data)
        if 'location' in input_dict:
            # This needs to match the location encoding from training
            # For now, we'll use a placeholder; in a real implementation,
            # you would need to have the proper label encoder from training
            mapped['kecamatan'] = str(input_dict['location'])
        else:
            # Placeholder - you need to handle this based on your training data
            mapped['kecamatan'] = "default"

        # Map previous crop to nama_tanaman, but we'll remove it before prediction
        # since it's the target variable
        if 'previous_crop' in input_dict:
            mapped['nama_tanaman'] = str(input_dict['previous_crop'])
        else:
            mapped['nama_tanaman'] = "default"

        # For the remaining fields, we need default values from training statistics
        # Add default values for other required fields based on training data
        # These should be set to average values from the training dataset
        if 'moisture' not in mapped:
            mapped['moisture'] = 0.0  # Default value from training normalization
        if 'fertility' not in mapped:
            mapped['fertility'] = 0.0  # Default value from training normalization
        if 'sunlight' not in mapped:
            mapped['sunlight'] = 0.0  # Default value from training normalization

        return mapped

    def predict(self,
               input_data: Union[pd.DataFrame, np.ndarray, dict, List[dict]],
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

    def predict_with_recommendations(self, input_data: Union[Dict, List[Dict]]) -> Union[List[Dict], Dict]:
        """
        Predict with detailed recommendations in the specified format

        Args:
            input_data: Input data in the format {"soil_ph": float, "temperature": float, ...}

        Returns:
            Recommendations in the format:
            [{"nama_tanaman":"Jagung Hibrida","kecocokan":0.65,"keterangan":"..."}]
        """
        with torch.no_grad():
            # Preprocess input
            X_encoded, feature_names = self.preprocess_input(input_data)

            # Convert to tensor
            X_tensor = torch.FloatTensor(X_encoded).to(self.device)

            # Forward pass
            output = self.model(X_tensor, memory_mask=None)

            # Extract target predictions
            input_sizes = self.metadata['input_sizes']
            feature_sizes_sum = sum(input_sizes[:-1])  # All feature sizes before target
            target_size = input_sizes[-1]  # Size of target
            target_start = feature_sizes_sum
            target_end = target_start + target_size

            # Extract target predictions (logits)
            target_predictions = output[:, target_start:target_end]

            # Convert logits to probabilities
            probabilities = torch.softmax(target_predictions, dim=1).cpu().numpy()

            # Check if it's a single prediction
            single_input = isinstance(input_data, dict)
            if single_input:
                # Only one input, return single result
                result = self._generate_recommendation(probabilities[0])
                return result
            else:
                # Multiple inputs, return list of results
                results = []
                for prob in probabilities:
                    results.append(self._generate_recommendation(prob))
                return results

    def _generate_recommendation(self, probabilities: np.ndarray) -> List[Dict]:
        """
        Generate recommendations based on probabilities

        Args:
            probabilities: Array of probabilities for each class

        Returns:
            List of crop recommendations with suitability scores
        """
        recommendations = []

        # Get the crop names and their corresponding probabilities
        for class_idx, prob in enumerate(probabilities):
            if prob > 0.1:  # Only include crops with significant probability
                crop_name = self.crop_mapping.get(class_idx, f"Unknown Crop {class_idx}")

                # Add description based on crop type
                if crop_name == "Jagung":
                    description = "Dapat beradaptasi pada suhu lebih tinggi, perlu cek kesuburan NPK."
                elif crop_name == "Padi":
                    description = "Membutuhkan kelembaban tanah yang cukup, perhatikan drainase."
                elif crop_name == "Kedelai":
                    description = "Cocok untuk tanah dengan pH netral, butuh sinar matahari cukup."
                else:
                    description = "Rekomendasi tanaman berdasarkan kondisi tanah saat ini."

                recommendations.append({
                    "nama_tanaman": crop_name,
                    "kecocokan": float(prob),
                    "keterangan": description
                })

        # Sort by suitability score in descending order
        recommendations.sort(key=lambda x: x["kecocokan"], reverse=True)
        return recommendations

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