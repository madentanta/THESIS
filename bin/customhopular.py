"""
Hopular: Modern Hopfield Networks for Tabular Data
A faithful PyTorch implementation for custom CSV datasets

Based on the original ml-jku/hopular implementation
Paper: "Hopular: Modern Hopfield Networks for Tabular Data" (https://arxiv.org/abs/2206.00664)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class Hopfield(nn.Module):
    """
    Simplified Modern Hopfield Network (continuous associative memory)
    Based on hflayers.Hopfield with key components
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_heads: int = 1,
                 scaling: float = None,
                 dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = scaling if scaling is not None else 1.0 / math.sqrt(self.head_dim)

        # Query, Key, Value projections
        self.q_proj = nn.Linear(input_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(input_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(input_size, hidden_size, bias=True)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, input_size, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                association_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: (stored_pattern, state_pattern, stored_pattern_value)
                - stored_pattern: memory (batch, mem_len, input_size)
                - state_pattern: query (batch, query_len, input_size)
                - stored_pattern_value: values (batch, mem_len, input_size)
            association_mask: mask (batch, query_len, mem_len) or (batch, mem_len)
        Returns:
            Retrieved patterns (batch, query_len, input_size)
        """
        stored_pattern, state_pattern, stored_pattern_value = inputs
        batch_size = state_pattern.size(0)
        query_len = state_pattern.size(1)

        # Project to multi-head space
        Q = self.q_proj(state_pattern).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(stored_pattern).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(stored_pattern_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (modern Hopfield energy)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        # Apply mask if provided
        if association_mask is not None:
            batch_size_from_scores = scores.shape[0]  # This is the actual batch dimension in scores (after reshaping)
            query_len = scores.shape[2]  # This is query_len in scores

            if association_mask.dim() == 2:
                # (original_batch_size, mem_len) -> (batch_size_from_scores, 1, 1, mem_len)
                # Need to make sure it's broadcastable with scores shape (batch_size_from_scores, num_heads, query_len, mem_len)
                association_mask = association_mask.unsqueeze(1).unsqueeze(2)
                # This gives (original_batch_size, 1, 1, mem_len)
                # But scores has (batch_size_from_scores, num_heads, query_len, mem_len)
                # Since batch_size_from_scores = 1 (from unsqueeze(0)), we can broadcast
            elif association_mask.dim() == 3:
                # Could be (1, original_batch_size, mem_len) or (batch, query_len, mem_len)
                if association_mask.shape[0] == batch_size_from_scores and association_mask.shape[2] == scores.shape[3]:
                    # It's already (batch_size_from_scores, query_len, mem_len) format
                    association_mask = association_mask.unsqueeze(1)  # (batch_size_from_scores, 1, query_len, mem_len)
                else:
                    # It's (1, original_batch_size, mem_len), but we need to broadcast to match query_len
                    # Expand it to (1, original_batch_size, query_len, mem_len)
                    association_mask = association_mask.unsqueeze(2).expand(-1, -1, query_len, -1)
                    # Then unsqueeze for heads: (1, 1, original_batch_size, query_len, mem_len) -> (1, 1, query_len, mem_len)
                    association_mask = association_mask.unsqueeze(1)
            scores = scores.masked_fill(association_mask, float('-inf'))

        # Softmax retrieval
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Retrieve patterns
        retrieved = torch.matmul(attention, V)
        retrieved = retrieved.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        # Output projection
        output = self.out_proj(retrieved)

        return output


class EmbeddingBlock(nn.Module):
    """
    Block responsible for embedding an input sample in Hopular.
    Embeds each feature separately with position and type encodings.
    """

    def __init__(self,
                 input_sizes: List[int],
                 feature_size: int,
                 feature_discrete: Optional[torch.Tensor],
                 dropout_probability: float):
        super().__init__()
        self.input_sizes = input_sizes
        self.feature_size = feature_size
        self.feature_discrete = feature_discrete
        self.dropout_probability = dropout_probability

        # Compute feature boundaries for slicing
        self.feature_boundaries = torch.cumsum(torch.as_tensor([0] + input_sizes), dim=0)
        self.feature_boundaries = (self.feature_boundaries[:-1], self.feature_boundaries[1:])

        # Feature-specific embeddings (one linear layer per feature)
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(in_features=input_size, out_features=feature_size, bias=True)
            for input_size in input_sizes
        ])

        # Feature type embeddings (continuous=0, discrete=1)
        self.register_buffer('feature_types', torch.zeros(size=(len(input_sizes),), dtype=torch.long))
        if feature_discrete is not None:
            self.feature_types[feature_discrete] = 1
        self.feature_type_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=feature_size)

        # Feature position embeddings
        self.register_buffer('feature_positions', torch.arange(len(input_sizes), dtype=torch.long))
        self.feature_position_embeddings = nn.Embedding(num_embeddings=len(input_sizes), embedding_dim=feature_size)

        # Feature-specific output projections
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.GELU(),
                nn.Dropout(p=dropout_probability),
                nn.LayerNorm(normalized_shape=feature_size, elementwise_affine=True, eps=1e-12),
                nn.Linear(in_features=feature_size, out_features=feature_size, bias=True)
            )
            for _ in input_sizes
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Embed input samples.

        Args:
            input: (batch_size, sum(input_sizes))
        Returns:
            embedded: (batch_size, sum(feature_sizes)) = (batch_size, num_features * feature_size)
        """
        # Embed each feature separately
        feature_iterator = zip(self.feature_embeddings, self.feature_boundaries[0], self.feature_boundaries[1])
        input_embedded = torch.cat([
            feature_embedding(input[:, feature_begin:feature_end])
            for feature_embedding, feature_begin, feature_end in feature_iterator
        ], dim=1)

        # Add feature type and position embeddings
        batch_size = input.size(0)
        num_features = len(self.input_sizes)

        # Reshape to (batch, num_features, feature_size)
        input_embedded = input_embedded.view(batch_size, num_features, self.feature_size)

        # Add type embeddings
        type_emb = self.feature_type_embeddings(self.feature_types).unsqueeze(0)
        input_embedded = input_embedded + type_emb

        # Add position embeddings
        pos_emb = self.feature_position_embeddings(self.feature_positions).unsqueeze(0)
        input_embedded = input_embedded + pos_emb

        # Apply feature-specific projections
        output = torch.cat([
            self.feature_projections[i](input_embedded[:, i])
            for i in range(num_features)
        ], dim=1)

        return output


class SummarizationBlock(nn.Module):
    """
    Block responsible for summarizing the current prediction in Hopular.
    """

    def __init__(self,
                 input_sizes: List[int],
                 feature_size: int,
                 dropout_probability: float):
        super().__init__()
        self.input_sizes = input_sizes
        self.feature_size = feature_size

        # Total input size = num_features * feature_size
        input_dim = feature_size * len(input_sizes)
        output_dim = sum(input_sizes)

        self.feature_summarizations = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=dropout_probability),
            nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True, eps=1e-12),
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (batch_size, num_features * feature_size)
        Returns:
            output: (batch_size, sum(input_sizes))
        """
        return self.feature_summarizations(input)


class HopfieldBlock(nn.Module):
    """
    Block responsible for memory lookup operations in Hopular.
    """

    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 hidden_size: int,
                 num_heads: int,
                 scaling_factor: float,
                 dropout_probability: float,
                 normalize: bool):
        super().__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.num_features = input_size // feature_size
        assert (self.num_features * feature_size) == input_size

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.scaling_factor = scaling_factor

        # Modern Hopfield network for pattern retrieval
        self.hopfield_lookup = Hopfield(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            scaling=scaling_factor / math.sqrt(hidden_size),
            dropout=dropout_probability
        )

    def forward(self,
                input: torch.Tensor,
                memory: torch.Tensor,
                memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Perform memory lookup using modern Hopfield networks.

        Args:
            input: current prediction (batch, input_size) or (1, batch, input_size)
            memory: external memory (1, mem_size, input_size)
            memory_mask: mask (batch, mem_size)
        Returns:
            refined prediction (same shape as input)
        """
        # Ensure correct shapes
        if input.dim() == 2:
            input = input.unsqueeze(0)
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)

        # Retrieve patterns using modern Hopfield network
        retrieved_patterns = self.hopfield_lookup(
            (memory, input, memory),
            association_mask=memory_mask
        )

        # Residual connection
        return input + retrieved_patterns


class HopularBlock(nn.Module):
    """
    Block responsible for iteratively refining the current prediction in Hopular.
    Combines sample-sample and feature-feature associations.
    """

    def __init__(self,
                 input_size: int,
                 feature_size: int,
                 hidden_size: int,
                 hidden_size_factor: float,
                 num_heads: int,
                 scaling_factor: float,
                 dropout_probability: float):
        super().__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.hidden_size_factor = hidden_size_factor

        # Compute hidden sizes
        self.hidden_size_sample = hidden_size if hidden_size > 0 else max(1, input_size // num_heads)
        self.hidden_size_feature = hidden_size if hidden_size > 0 else max(1, feature_size // num_heads)

        assert input_size % feature_size == 0

        # Sample-sample associations (across samples in the batch)
        self.sample_norm = nn.LayerNorm(normalized_shape=input_size, elementwise_affine=True, eps=1e-12)
        self.sample_sample_associations = HopfieldBlock(
            input_size=input_size,
            feature_size=feature_size,
            hidden_size=max(1, int(self.hidden_size_sample * hidden_size_factor)),
            num_heads=num_heads,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            normalize=False
        )

        # Feature-feature associations (across features within a sample)
        self.feature_norm = nn.LayerNorm(normalized_shape=feature_size, elementwise_affine=True, eps=1e-12)
        self.feature_feature_associations = HopfieldBlock(
            input_size=feature_size,
            feature_size=feature_size,
            hidden_size=max(1, int(self.hidden_size_feature * hidden_size_factor)),
            num_heads=num_heads,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            normalize=False
        )

    def forward(self,
                input: torch.Tensor,
                sample_memory: torch.Tensor,
                sample_memory_mask: Optional[torch.Tensor],
                feature_memory: torch.Tensor,
                feature_memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Perform a single prediction refinement step.

        Args:
            input: current prediction (1, batch, input_size)
            sample_memory: training set memory (1, mem_size, input_size)
            sample_memory_mask: mask for sample lookups (batch, mem_size)
            feature_memory: current input representation (1, batch, input_size)
            feature_memory_mask: mask for feature lookups
        Returns:
            refined prediction (1, batch, input_size)
        """
        # Sample-sample interactions
        interactions = self.sample_sample_associations(
            self.sample_norm(input),
            memory=self.sample_norm(sample_memory),
            memory_mask=sample_memory_mask
        )

        # Reshape for feature-feature interactions
        # (1, batch, input_size) -> (batch, num_features, feature_size)
        batch_size = interactions.shape[1]
        num_features = self.input_size // self.feature_size
        interactions = interactions.reshape(batch_size, num_features, self.feature_size)
        feature_memory_reshaped = feature_memory.reshape(batch_size, num_features, self.feature_size)

        # Feature-feature interactions (process each sample's features)
        # Transpose to (1, batch*num_features, feature_size) for Hopfield
        interactions = interactions.reshape(1, batch_size * num_features, self.feature_size)
        feature_memory_reshaped = feature_memory_reshaped.reshape(1, batch_size * num_features, self.feature_size)

        interactions = self.feature_feature_associations(
            self.feature_norm(interactions),
            memory=self.feature_norm(feature_memory_reshaped),
            memory_mask=feature_memory_mask
        )

        # Reshape back to original shape
        interactions = interactions.reshape(1, batch_size, self.input_size)

        return interactions


class Hopular(nn.Module):
    """
    Implementation of Hopular: Modern Hopfield Networks for Tabular Data
    """

    def __init__(self,
                 input_sizes: List[int],
                 target_discrete: List[int],
                 target_numeric: List[int],
                 feature_discrete: Optional[List[int]],
                 memory: torch.Tensor,
                 memory_ids: Optional[torch.Tensor] = None,
                 feature_size: int = 32,
                 hidden_size: int = 32,
                 hidden_size_factor: float = 1.0,
                 num_heads: int = 8,
                 scaling_factor: float = 1.0,
                 input_dropout: float = 0.1,
                 lookup_dropout: float = 0.1,
                 output_dropout: float = 0.1,
                 memory_ratio: float = 1.0,
                 num_blocks: int = 4):
        super().__init__()

        self.input_sizes = input_sizes
        self.feature_size = feature_size
        self.target_discrete = target_discrete
        self.target_numeric = target_numeric
        self.feature_discrete = feature_discrete
        self.num_blocks = num_blocks
        self.memory_ratio = memory_ratio

        # Compute total input size
        self.input_size = feature_size * len(input_sizes)

        # Feature boundaries for output reconstruction
        self.feature_boundaries = torch.cumsum(torch.as_tensor([0] + input_sizes), dim=0)
        self.feature_boundaries = (self.feature_boundaries[:-1], self.feature_boundaries[1:])

        # Embedding block
        feature_discrete_tensor = torch.tensor(feature_discrete) if feature_discrete else None
        self.embeddings = EmbeddingBlock(
            input_sizes=input_sizes,
            feature_size=feature_size,
            feature_discrete=feature_discrete_tensor,
            dropout_probability=input_dropout
        )

        # Memory (training set)
        self.register_buffer('memory', memory)
        self.register_buffer('memory_ids',
                           torch.arange(len(memory)) if memory_ids is None else memory_ids)

        # Hopular blocks
        self.hopular_blocks = nn.ModuleList([
            HopularBlock(
                input_size=self.input_size,
                feature_size=feature_size,
                hidden_size=hidden_size,
                hidden_size_factor=hidden_size_factor,
                num_heads=num_heads,
                scaling_factor=scaling_factor,
                dropout_probability=lookup_dropout
            )
            for _ in range(num_blocks)
        ])

        # Summarization block
        self.summarizations = SummarizationBlock(
            input_sizes=input_sizes,
            feature_size=feature_size,
            dropout_probability=output_dropout
        )

    def forward(self,
                input: torch.Tensor,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply Hopular.

        Args:
            input: input samples (batch_size, sum(input_sizes))
            memory_mask: mask for memory lookups (batch_size, memory_size)
        Returns:
            predictions (batch_size, sum(input_sizes))
        """
        # Embed input and memory
        embeddings = self.embeddings(input).unsqueeze(0)  # (1, batch, input_size)
        embeddings_memory = self.embeddings(self.memory).unsqueeze(0)  # (1, mem_size, input_size)

        # Optionally subsample memory during training
        iteration_memory = embeddings_memory
        iteration_mask = memory_mask

        if self.training and self.memory_ratio < 1.0:
            memory_indices = torch.randperm(embeddings_memory.shape[1], device=embeddings_memory.device)
            memory_indices = memory_indices[:max(1, int(self.memory_ratio * memory_indices.shape[0]))]
            iteration_memory = iteration_memory[:, memory_indices]
            if iteration_mask is not None:
                iteration_mask = iteration_mask[:, memory_indices]

        # Apply Hopular blocks (iterative refinement)
        hopular_iteration = embeddings
        for hopular_block in self.hopular_blocks:
            # Adjust memory mask to match reshaped input dimensions
            # When hopular_iteration is unsqueezed to (1, batch_size, input_size),
            # the mask should also be adjusted from (batch_size, memory_size) to (1, batch_size, memory_size)
            # Actually, in the Hopfield attention, we need the mask as (batch, query_len, mem_len)
            # So with state_pattern having shape (1, 3, 288) -> batch_size=1, query_len=3
            # and stored_pattern having shape (1, 9, 288) -> mem_len=9
            # The mask should be (1, 3, 9) if it was 3D, but it's 2D (3, 9)
            # So we need to expand the mask to account for the new batch dimension
            adjusted_mask = iteration_mask
            if adjusted_mask is not None:
                # Original mask has shape (original_batch_size, memory_size)
                # Hopular embeddings have been reshaped to (1, original_batch_size, input_size)
                # So mask should become (1, original_batch_size, memory_size)
                adjusted_mask = adjusted_mask.unsqueeze(0)  # (1, original_batch_size, memory_size)

            hopular_iteration = hopular_block(
                hopular_iteration,
                sample_memory=iteration_memory,
                sample_memory_mask=adjusted_mask,
                feature_memory=embeddings,
                feature_memory_mask=None
            )

        # Summarize to output
        hopular_iteration = hopular_iteration.reshape(hopular_iteration.shape[1], -1)
        return self.summarizations(hopular_iteration)


class TabularDataset(Dataset):
    """Dataset wrapper for tabular data with masking support"""

    def __init__(self, X: np.ndarray, y: np.ndarray, sizes: List[int],
                 target_indices: List[int], mask_prob: float = 0.15):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y.dtype == np.float32 else torch.LongTensor(y)
        self.sizes = sizes
        self.target_indices = target_indices
        self.mask_prob = mask_prob

        # Compute feature boundaries
        self.boundaries = torch.cumsum(torch.tensor([0] + sizes), dim=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y_target = self.y[idx]

        # Create masked version for training
        x_masked = x.clone()
        mask = torch.zeros(len(self.sizes), dtype=torch.bool)

        if self.training:
            # Randomly mask features (BERT-style)
            for i in range(len(self.sizes)):
                if i not in self.target_indices and torch.rand(1) < self.mask_prob:
                    start, end = self.boundaries[i], self.boundaries[i+1]
                    x_masked[start:end] = 0
                    mask[i] = True

        # Create missing data indicator (all zeros for complete data)
        missing = torch.zeros(len(self.sizes), dtype=torch.bool)

        return x_masked, mask, x, missing, torch.tensor(idx, dtype=torch.long)



def remove_rare_classes(X: np.ndarray, y: np.ndarray, min_samples: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove classes with fewer than min_samples samples

    Args:
        X: features
        y: targets
        min_samples: minimum number of samples per class
    Returns:
        X_filtered, y_filtered
    """
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= min_samples]

    if len(valid_classes) < len(unique):
        removed_classes = unique[counts < min_samples]
        print(f"Warning: Removing {len(removed_classes)} class(es) with < {min_samples} samples")
        print(f"Removed classes: {removed_classes}")

        mask = np.isin(y, valid_classes)
        return X[mask], y[mask]

    return X, y


def load_and_preprocess_csv(csv_path: str,
                            target_column: str,
                            categorical_columns: Optional[List[str]] = None,
                            test_size: float = 0.2,
                            random_state: int = 42,
                            min_class_samples: int = 3) -> Tuple:
    """
    Load and preprocess CSV data for Hopular

    Args:
        csv_path: Path to CSV file
        target_column: Name of target column
        categorical_columns: List of categorical column names (auto-detect if None)
        test_size: Proportion of data for test+val (will be split again)
        random_state: Random seed
        min_class_samples: Minimum samples per class (removes rare classes)

    Returns:
        train_dataset, val_dataset, test_dataset, metadata dictionary
    """
    # Load data
    df = pd.read_csv(csv_path)

    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)-1} features")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    original_features = list(X.columns)

    # Identify categorical columns
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Encode categorical features
    label_encoders = {}
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Determine task type and encode target
    task = 'classification'
    target_label_encoder = None

    if y.dtype == 'object' or len(y.unique()) < 20:
        target_label_encoder = LabelEncoder()
        y = target_label_encoder.fit_transform(y)
        task = 'classification'
        print(f"Task: Classification with {len(np.unique(y))} classes")

        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
    else:
        task = 'regression'
        y = y.values.astype(np.float32)
        print(f"Task: Regression")

    X_values = X.values.astype(np.float32)

    # For classification, remove rare classes if needed
    if task == 'classification':
        X_values, y = remove_rare_classes(X_values, y, min_samples=min_class_samples)
        print(f"After filtering: {len(X_values)} samples")

        # Re-encode the target labels to ensure they are consecutive from 0 to n_classes-1
        # This fixes the issue where class values after filtering might not be consecutive
        unique_labels = np.unique(y)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

        # Apply the mapping to re-encode all the target values
        y = np.array([label_map[old_label] for old_label in y])

    scaler = StandardScaler()
    scaler.fit(X_values)
    X_values = scaler.transform(X_values).astype(np.float32)

    # Check if stratification is possible for classification
    stratify_train = None
    stratify_val = None
    if task == 'classification':
        # Check if all classes have at least 2 samples
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()

        if min_samples >= 2 and len(X_values) >= 10:
            stratify_train = y
            print(f"Using stratified split (min class size: {min_samples})")
        else:
            print(f"Warning: Some classes have only {min_samples} sample(s).")
            print("Using random split instead of stratified split.")
            stratify_train = None

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_values, y, test_size=test_size * 2, random_state=random_state,
        stratify=stratify_train
    )

    # Check stratification for validation split
    if task == 'classification' and stratify_train is not None:
        unique, counts = np.unique(y_temp, return_counts=True)
        if counts.min() >= 2:
            stratify_val = y_temp
        else:
            stratify_val = None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state,
        stratify=stratify_val
    )

    # Note: y, y_train, y_val, y_test are already re-encoded at this point due to the earlier steps
    # so no further re-encoding is needed


    # Prepare sizes for Hopular (each feature gets size 1 for continuous, or n_classes for categorical)
    input_sizes = []
    feature_discrete = []

    for i, col in enumerate(X.columns):
        if col in categorical_columns:
            n_classes = len(label_encoders[col].classes_)
            input_sizes.append(n_classes)
            feature_discrete.append(i)
        else:
            input_sizes.append(1)

    # One-hot encode features for Hopular
    X_train_encoded = encode_features(X_train, input_sizes, feature_discrete)
    X_val_encoded = encode_features(X_val, input_sizes, feature_discrete)
    X_test_encoded = encode_features(X_test, input_sizes, feature_discrete)

    # One-hot encode targets if classification
    if task == 'classification':
        # Recalculate n_classes after removing rare classes and re-encoding
        n_classes = len(np.unique(y))
        y_train_encoded = F.one_hot(torch.LongTensor(y_train), num_classes=n_classes).float().numpy()
        y_val_encoded = F.one_hot(torch.LongTensor(y_val), num_classes=n_classes).float().numpy()
        y_test_encoded = F.one_hot(torch.LongTensor(y_test), num_classes=n_classes).float().numpy()

        target_discrete = [len(input_sizes)]  # Target is the last "feature"
        target_numeric = []
        input_sizes.append(n_classes)
    else:
        y_train_encoded = y_train.reshape(-1, 1)
        y_val_encoded = y_val.reshape(-1, 1)
        y_test_encoded = y_test.reshape(-1, 1)

        target_discrete = []
        target_numeric = [len(input_sizes)]
        input_sizes.append(1)

    # Concatenate features and targets
    X_train_full = np.concatenate([X_train_encoded, y_train_encoded], axis=1)
    X_val_full = np.concatenate([X_val_encoded, y_val_encoded], axis=1)
    X_test_full = np.concatenate([X_test_encoded, y_test_encoded], axis=1)

    metadata = {
        'input_sizes': input_sizes,
        'feature_discrete': feature_discrete,
        'target_discrete': target_discrete,
        'target_numeric': target_numeric,
        'task': task,
        'n_features': len(X.columns),
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_label_encoder': target_label_encoder,
        'memory': torch.FloatTensor(X_train_full),
        'original_features': list(X.columns)
    }

    return X_train_full, X_val_full, X_test_full, y_train, y_val, y_test, metadata

def analyze_dataset(csv_path: str, target_column: str) -> None:
    """
    Analyze dataset and print useful statistics

    Args:
        csv_path: Path to CSV file
        target_column: Name of target column
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(f"\n📊 Basic Info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Total features: {len(df.columns) - 1}")
    print(f"  Target column: '{target_column}'")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"   {col}: {count} ({100*count/len(df):.1f}%)")
    else:
        print(f"\n✔️ No missing values")

    # Analyze target
    y = df[target_column]
    if y.dtype == object or len(y.unique()) < 20:
        print(f"\n🎯 Target (Classification):")
        print(f"  Number of classes: {len(y.unique())}")
        print(f"\n  Class distribution:")
        counts = y.value_counts().sort_index()
        for class_val, count in counts.items():
            print(f"   {class_val}: {count} samples ({100*count/len(df):.1f}%)")

        # Check for rare classes
        min_count = counts.min()
        if min_count < 3:
            print(f"\n⚠️  WARNING: Smallest class has only {min_count} sample(s)")
            rare_classes = counts[counts < 3].index.tolist()
            print(f"  Rare classes: {rare_classes}")
        elif min_count < 10:
            print(f"\nℹ️  Note: Smallest class has {min_count} samples")
    else:
        print(f"\n🎯 Target (Regression):")
        print(f"  Min: {y.min():.2f}")
        print(f"  Max: {y.max():.2f}")
        print(f"  Mean: {y.mean():.2f}")
        print(f"  Std: {y.std():.2f}")

    # Analyze features
    X = df.drop(columns=[target_column])
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n🔍 Feature Types:")
    print(f"  Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

    print("\n✔️ Dataset analysis complete.")



def encode_features(X: np.ndarray, input_sizes: List[int], feature_discrete: List[int]) -> np.ndarray:
    """One-hot encode categorical features"""
    encoded_features = []

    for i, size in enumerate(input_sizes):
        if i in feature_discrete:
            # One-hot encode with safety check for out-of-bounds values
            feature_vals = X[:, i].astype(int)

            # Check if any values are out of bounds (>= size) and clip them
            if np.any(feature_vals >= size):
                print(f"Warning: Found values >= {size} for feature {i}, clipping to {size-1}")
                feature_vals = np.clip(feature_vals, 0, size-1)

            one_hot = F.one_hot(torch.LongTensor(feature_vals), num_classes=size).float().numpy()
            encoded_features.append(one_hot)
        else:
            # Keep as is (already standardized)
            encoded_features.append(X[:, i:i+1])

    return np.concatenate(encoded_features, axis=1)


# Example usage
if __name__ == "__main__":
    print("Hopular implementation for CSV datasets")
    print("\n1. First, analyze your dataset:")
    print("   analyze_dataset('your_data.csv', 'target_column')")
    print("\n2. Then load and preprocess:")
    print("   X_train, X_val, X_test, y_train, y_val, y_test, metadata = \\")
    print("       load_and_preprocess_csv('your_data.csv', 'target_column')")
    print("\n3. See hopular_trainer.py for training examples")
    print("\nQuick test with Iris dataset:")

    try:
        from sklearn.datasets import load_iris
        import pandas as pd

        # Create iris CSV
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df.to_csv('/tmp/iris.csv', index=False)

        print("\n" + "="*60)
        analyze_dataset('/tmp/iris.csv', 'target')
        print("\nIris dataset analyzed successfully!")
        print("Try: load_and_preprocess_csv('/tmp/iris.csv', 'target')")
    except Exception as e:
        print(f"\nCouldn't create test dataset: {e}")
        print("Use your own CSV file instead.")