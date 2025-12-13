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
import numpy as np
from torch.utils.data import Dataset
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

        # Add a target-specific output layer to ensure better target prediction
        # We'll identify the target position and size to apply specific processing
        if target_discrete:
            target_idx = target_discrete[0]  # First target index
            target_size = input_sizes[target_idx]
            self.target_output_layer = nn.Sequential(
                nn.Linear(feature_size, feature_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_size // 2, target_size),
                nn.LogSoftmax(dim=-1)  # Ensure proper probability distribution for classification
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
        output = self.summarizations(hopular_iteration)

        return output


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


def encode_features(X: np.ndarray, input_sizes: List[int],
                   feature_discrete: List[int]) -> np.ndarray:
    """One-hot encode categorical features"""
    encoded_features = []
    col_idx = 0

    for i, size in enumerate(input_sizes):
        if i in feature_discrete:
            # One-hot encode
            feature_vals = X[:, col_idx].astype(int)
            feature_vals = np.clip(feature_vals, 0, size-1)
            one_hot = F.one_hot(torch.LongTensor(feature_vals),
                               num_classes=size).float().numpy()
            encoded_features.append(one_hot)
        else:
            # Keep continuous as-is
            encoded_features.append(X[:, col_idx:col_idx+1])

        col_idx += 1

    return np.concatenate(encoded_features, axis=1)