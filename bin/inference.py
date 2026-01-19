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
                 model_path: str = 'output/best_hopular_model.pt',
                 metadata_path: str = 'output/metadata.pkl',
                 device: str = None):
        """
        Initialize the inference module
        """
        # Load preprocessing metadata
        preprocess_meta_path = "data/preprocess_metadata.pkl"
        if os.path.exists(preprocess_meta_path):
            with open(preprocess_meta_path, "rb") as f:
                self.preprocess_metadata = pickle.load(f)
        else:
            raise FileNotFoundError(f"Preprocessing metadata {preprocess_meta_path} not found.")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

        # Load the trained model
        self.model = self._load_model(model_path)
        self.model.eval()

        print(f"Hopular inference module loaded on {self.device}")

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

    def _map_input_format(self, input_dict: dict) -> dict:
        """Mapping fields from Laravel format to Model format"""
        mapped = {}
        mapped['ph'] = float(input_dict.get('soil_ph', input_dict.get('ph', 0.0)))
        mapped['temp'] = float(input_dict.get('temperature', input_dict.get('temp', 0.0)))
        mapped['humidity'] = float(input_dict.get('humidity', 0.0))
        mapped['kecamatan'] = str(input_dict.get('location', "default"))
        mapped['nama_tanaman'] = str(input_dict.get('previous_crop', "default"))
        # Fields default untuk melengkapi input tensor
        mapped['moisture'] = 0.0
        mapped['fertility'] = 0.0
        mapped['sunlight'] = 0.0
        return mapped

    def preprocess_input(self, input_data):
        """Standardize input for the model"""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([self._map_input_format(input_data)])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame([self._map_input_format(x) for x in input_data])
        else:
            raise ValueError("Unsupported input type")

        meta = self.preprocess_metadata
        feature_names = meta["feature_names"]
        continuous_cols = meta["continuous_cols"]
        categorical_cols = meta["categorical_cols"]

        # Ensure all columns exist
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0

        input_df = input_df[feature_names]

        # Categorical encoding
        for col in categorical_cols:
            le = meta["label_encoders"][col]
            known = set(le.classes_)
            default = le.classes_[0]
            input_df[col] = input_df[col].astype(str).apply(lambda x: x if x in known else default)
            input_df[col] = le.transform(input_df[col])

        # Numerical scaling
        scaler = meta["scaler"]
        input_df[continuous_cols] = scaler.transform(input_df[continuous_cols].astype(np.float32))

        X = input_df.values.astype(np.float32)
        n_features = len(feature_names)
        input_sizes = self.metadata["input_sizes"][:n_features]
        feature_discrete = [i for i in self.metadata["feature_discrete"] if i < n_features]

        X_encoded = encode_features(X, input_sizes, feature_discrete)

        # Pad to full input size if necessary
        total_size = sum(self.metadata["input_sizes"])
        X_full = np.zeros((X_encoded.shape[0], total_size), dtype=np.float32)
        X_full[:, :X_encoded.shape[1]] = X_encoded

        return X_full, feature_names

    def predict_with_recommendations(self, input_data: Union[Dict, List[Dict]]) -> Union[List[Dict], Dict]:
        """Run inference and return top-1 recommendation for each input"""
        with torch.no_grad():
            X_encoded, _ = self.preprocess_input(input_data)
            X_tensor = torch.FloatTensor(X_encoded).to(self.device)
            output = self.model(X_tensor, memory_mask=None)

            input_sizes = self.metadata['input_sizes']
            feature_sizes_sum = sum(input_sizes[:-1])
            target_size = input_sizes[-1]
            
            target_predictions = output[:, feature_sizes_sum:feature_sizes_sum + target_size]
            probabilities = torch.softmax(target_predictions, dim=1).cpu().numpy()

            if isinstance(input_data, dict):
                return self._generate_recommendation(probabilities[0])
            else:
                return [self._generate_recommendation(prob) for prob in probabilities]

    def _generate_recommendation(self, probabilities: np.ndarray) -> Dict:
        """LOGIC: Sort all possibilities and return only the TOP 1"""
        target_encoder = self.metadata.get('target_label_encoder')
        class_names = target_encoder.classes_ if target_encoder else ["Tebu", "Jagung", "Padi"]

        # Create list of all possible crops with their scores
        recommendations = [
            {"nama_tanaman": class_names[i], "kecocokan": float(prob)}
            for i, prob in enumerate(probabilities)
        ]

        # SORTING: Urutkan dari skor tertinggi ke terendah
        recommendations.sort(key=lambda x: x['kecocokan'], reverse=True)

        # RETURN TOP 1 ONLY
        # Meskipun ada 3 yang di atas 0.5, kita hanya ambil yang paling tinggi (index 0)
        return recommendations[0]

    def _postprocess_predictions(self, target_predictions: torch.Tensor) -> np.ndarray:
        """Standard argmax prediction"""
        predictions = torch.argmax(target_predictions, dim=1).cpu().numpy()
        target_encoder = self.metadata.get('target_label_encoder')
        if target_encoder:
            predictions = target_encoder.inverse_transform(predictions)
        return predictions