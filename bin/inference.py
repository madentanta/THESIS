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
        """
        with open("data/preprocess_metadata.pkl", "rb") as f:
            self.preprocess_metadata = pickle.load(f)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

        # Load the trained model
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"Hopular inference module loaded on {self.device}")
        print(f"Task: {self.metadata.get('task', 'Unknown')}")

    def _load_model(self, model_path: str) -> Hopular:
        """Load the trained Hopular model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        model = Hopular(
            input_sizes=self.metadata['input_sizes'],
            target_discrete=self.metadata['target_discrete'],
            target_numeric=self.metadata['target_numeric'],
            feature_discrete=self.metadata['feature_discrete'],
            memory=self.metadata['memory'],
            feature_size=32,
            hidden_size=32,
            hidden_size_factor=1.0,
            num_heads=4,
            num_blocks=2,
            scaling_factor=1.0,
            input_dropout=0.1,
            lookup_dropout=0.1,
            output_dropout=0.1,
            memory_ratio=1.0
        )

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)

        return model

    def preprocess_input(self, input_data):
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([self._map_input_format(input_data)])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame([self._map_input_format(x) for x in input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Unsupported input type")

        meta = self.preprocess_metadata
        feature_names = meta["feature_names"]
        continuous_cols = meta["continuous_cols"]
        categorical_cols = meta["categorical_cols"]

        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0

        input_df = input_df[feature_names]

        for col in categorical_cols:
            le = meta["label_encoders"][col]
            known = set(le.classes_)
            default = le.classes_[0]

            input_df[col] = input_df[col].astype(str)
            input_df[col] = input_df[col].apply(lambda x: x if x in known else default)
            input_df[col] = le.transform(input_df[col])

        scaler = meta["scaler"]
        input_df[continuous_cols] = scaler.transform(input_df[continuous_cols].astype(np.float32))

        X = input_df.values.astype(np.float32)

        n_features = len(feature_names)
        input_sizes = self.metadata["input_sizes"][:n_features]
        feature_discrete = [i for i in self.metadata["feature_discrete"] if i < n_features]

        X_encoded = encode_features(X, input_sizes, feature_discrete)

        total_size = sum(self.metadata["input_sizes"])
        X_full = np.zeros((X_encoded.shape[0], total_size), dtype=np.float32)
        X_full[:, :sum(input_sizes)] = X_encoded

        return X_full, feature_names

    def _map_input_format(self, input_dict: dict) -> dict:
        mapped = {}
        mapped['ph'] = float(input_dict.get('soil_ph', input_dict.get('ph', 0.0)))
        mapped['temp'] = float(input_dict.get('temperature', input_dict.get('temp', 0.0)))
        mapped['humidity'] = float(input_dict.get('humidity', 0.0))
        mapped['kecamatan'] = str(input_dict.get('location', "default"))
        mapped['nama_tanaman'] = str(input_dict.get('previous_crop', "default"))
        mapped['moisture'] = 0.0
        mapped['fertility'] = 0.0
        mapped['sunlight'] = 0.0
        return mapped

    def predict(self,
                input_data: Union[pd.DataFrame, np.ndarray, dict, List[dict]],
                feature_columns: Optional[List[str]] = None,
                return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        with torch.no_grad():
            X_encoded, feature_names = self.preprocess_input(input_data)
            X_tensor = torch.FloatTensor(X_encoded).to(self.device)
            output = self.model(X_tensor, memory_mask=None)

            input_sizes = self.metadata['input_sizes']
            feature_sizes_sum = sum(input_sizes[:-1])
            target_size = input_sizes[-1]
            target_predictions = output[:, feature_sizes_sum:feature_sizes_sum + target_size]

            predictions = self._postprocess_predictions(target_predictions)

            if return_probabilities and self.metadata.get('target_discrete'):
                probabilities = torch.softmax(target_predictions, dim=1).cpu().numpy()
                return predictions, probabilities
            else:
                return predictions

    def predict_with_recommendations(self, input_data: Union[Dict, List[Dict]]) -> Union[List[Dict], Dict]:
        with torch.no_grad():
            X_encoded, _ = self.preprocess_input(input_data)
            X_tensor = torch.FloatTensor(X_encoded).to(self.device)
            output = self.model(X_tensor, memory_mask=None)

            input_sizes = self.metadata['input_sizes']
            feature_sizes_sum = sum(input_sizes[:-1])
            target_size = input_sizes[-1]
            target_predictions = output[:, feature_sizes_sum:feature_sizes_sum + target_size]
            probabilities = torch.softmax(target_predictions, dim=1).cpu().numpy()

            single_input = isinstance(input_data, dict)
            if single_input:
                return self._generate_recommendation(probabilities[0])
            else:
                return [self._generate_recommendation(prob) for prob in probabilities]

    def _generate_recommendation(self, probabilities: np.ndarray) -> Dict:
        """Return only top recommendation with probability >= 0.5"""
        target_encoder = self.metadata.get('target_label_encoder')
        class_names = target_encoder.classes_ if target_encoder else ["Tebu", "Jagung", "Padi"]

        # Filter probabilitas >= 0.5
        recommendations = [
            {"nama_tanaman": class_names[i], "kecocokan": float(prob)}
            for i, prob in enumerate(probabilities)
            if prob >= 0.5
        ]

        # Kalau ga ada >=0.5, ambil yang tertinggi
        if not recommendations:
            max_idx = int(np.argmax(probabilities))
            recommendations = [{"nama_tanaman": class_names[max_idx],
                                "kecocokan": float(probabilities[max_idx])}]

        # Ambil rekomendasi pertama (top)
        return recommendations[0]

    def _postprocess_predictions(self, target_predictions: torch.Tensor) -> np.ndarray:
        task = self.metadata.get('task', 'classification')
        if task == 'classification' or self.metadata.get('target_discrete'):
            predictions = torch.argmax(target_predictions, dim=1).cpu().numpy()
            target_encoder = self.metadata.get('target_label_encoder')
            if target_encoder:
                predictions = target_encoder.inverse_transform(predictions)
        else:
            predictions = target_predictions.cpu().numpy()
        return predictions

    def predict_from_file(self,
                         file_path: str,
                         target_column: str = None,
                         return_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        df = pd.read_csv(file_path)
        X_df = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df
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
    return HopularInference(model_path, metadata_path, device)


def predict_new_data(input_data: Union[pd.DataFrame, np.ndarray, dict],
                     model_path: str = 'best_hopular_model.pt',
                     metadata_path: str = 'metadata.pkl',
                     feature_columns: Optional[List[str]] = None,
                     device: str = None) -> np.ndarray:
    inference_model = load_trained_model(model_path, metadata_path, device)
    return inference_model.predict(input_data, feature_columns)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hopular Model Inference")
    parser.add_argument("--model_path", type=str, default="best_hopular_model.pt")
    parser.add_argument("--metadata_path", type=str, default="metadata.pkl")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--target_column", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="predictions.csv")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    hopular_inference = HopularInference(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        device=args.device
    )

    predictions_df = hopular_inference.predict_from_file(
        file_path=args.input_file,
        target_column=args.target_column,
        return_dataframe=True
    )

    predictions_df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")
