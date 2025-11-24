"""
Complete training script for Hopular on CSV datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm


class HopularTrainer:
    """
    Trainer for Hopular matching the original training procedure
    """

    def __init__(self,
                 model,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.01,
                 initial_feature_loss_weight: float = 1.0,
                 final_feature_loss_weight: float = 0.0):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.initial_feature_loss_weight = initial_feature_loss_weight
        self.final_feature_loss_weight = final_feature_loss_weight
        self.annealing_factor = initial_feature_loss_weight

        # Loss functions
        self.loss_numeric = nn.MSELoss(reduction='mean')
        self.loss_discrete = nn.CrossEntropyLoss(reduction='mean')

    def compute_performance(self,
                           result: torch.Tensor,
                           data_noise: torch.Tensor,
                           data_unmasked: torch.Tensor,
                           data_missing: torch.Tensor,
                           input_sizes: list,
                           feature_discrete: list,
                           target_discrete: list,
                           target_numeric: list) -> Dict[str, torch.Tensor]:
        """
        Compute losses and accuracies for features and targets
        """
        feature_boundaries = torch.cumsum(torch.tensor([0] + input_sizes), dim=0)

        feature_count = 0
        loss_feature = torch.zeros(1, device=result.device)
        accuracy_feature = torch.zeros(1, device=result.device)

        target_count = 0
        loss_target = torch.zeros(1, device=result.device)
        accuracy_target = torch.zeros(1, device=result.device)

        # Compute loss for each feature
        for feature_idx in range(len(input_sizes)):
            start, end = feature_boundaries[feature_idx], feature_boundaries[feature_idx + 1]

            # Check which samples have this feature masked and not missing
            data_valid = torch.logical_not(data_missing[:, feature_idx])
            data_valid = torch.logical_and(data_valid, data_noise[:, feature_idx])

            if not data_valid.any():
                continue

            prediction = result[data_valid, start:end]
            target = data_unmasked[data_valid, start:end]

            # Discrete feature (classification)
            if feature_idx in feature_discrete:
                loss = self.loss_discrete(prediction, target.argmax(dim=1))
                accuracy = (prediction.detach().argmax(dim=1) == target.detach().argmax(dim=1)).float().mean()

                if feature_idx in target_discrete:
                    loss_target = loss_target + loss
                    accuracy_target += accuracy
                    target_count += 1
                else:
                    loss_feature = loss_feature + loss
                    accuracy_feature += accuracy
                    feature_count += 1
            else:
                # Continuous feature (regression)
                loss = self.loss_numeric(prediction, target)

                if feature_idx in target_numeric:
                    loss_target = loss_target + loss
                    target_count += 1
                else:
                    loss_feature = loss_feature + loss
                    feature_count += 1

        # Average losses
        if feature_count > 0:
            loss_feature = loss_feature / feature_count
            accuracy_feature = accuracy_feature / feature_count

        if target_count > 0:
            loss_target = loss_target / target_count
            accuracy_target = accuracy_target / target_count

        return {
            'loss_feature': loss_feature,
            'loss_target': loss_target,
            'accuracy_feature': accuracy_feature,
            'accuracy_target': accuracy_target,
            'feature_count': feature_count,
            'target_count': target_count
        }

    def train_epoch(self,
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   metadata: Dict) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        total_feature_loss = 0
        total_target_loss = 0
        total_feature_acc = 0
        total_target_acc = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc='Training'):
            data_masked, data_noise, data_unmasked, data_missing, data_indices = batch

            data_masked = data_masked.to(self.device)
            data_noise = data_noise.to(self.device)
            data_unmasked = data_unmasked.to(self.device)
            data_missing = data_missing.to(self.device)
            data_indices = data_indices.to(self.device)

            # Create memory mask (prevent self-lookup)
            memory_mask = data_indices.view(-1, 1) == self.model.memory_ids.view(1, -1)

            # Forward pass
            optimizer.zero_grad()
            result = self.model(data_masked, memory_mask=memory_mask)

            # Compute performance
            perf = self.compute_performance(
                result, data_noise, data_unmasked, data_missing,
                metadata['input_sizes'],
                metadata['feature_discrete'],
                metadata['target_discrete'],
                metadata['target_numeric']
            )

            # Compute combined loss with annealing
            if perf['feature_count'] <= 0:
                loss = perf['loss_target']
                annealing_factor = 0.0
            elif perf['target_count'] <= 0:
                loss = perf['loss_feature']
                annealing_factor = 1.0
            else:
                annealing_factor = self.annealing_factor
                loss = annealing_factor * perf['loss_feature'] + (1.0 - annealing_factor) * perf['loss_target']

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_feature_loss += perf['loss_feature'].item()
            total_target_loss += perf['loss_target'].item()
            if perf['feature_count'] > 0:
                total_feature_acc += perf['accuracy_feature'].item()
            if perf['target_count'] > 0:
                total_target_acc += perf['accuracy_target'].item()
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'loss_feature': total_feature_loss / n_batches,
            'loss_target': total_target_loss / n_batches,
            'accuracy_feature': total_feature_acc / n_batches,
            'accuracy_target': total_target_acc / n_batches
        }

    def evaluate(self,
                val_loader: DataLoader,
                metadata: Dict) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()

        total_loss = 0
        total_accuracy = 0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                data_masked, data_noise, data_unmasked, data_missing = batch[:4]

                data_masked = data_masked.to(self.device)
                data_noise = data_noise.to(self.device)
                data_unmasked = data_unmasked.to(self.device)
                data_missing = data_missing.to(self.device)

                # Forward pass (no memory mask during evaluation)
                result = self.model(data_masked, memory_mask=None)

                # Compute performance
                perf = self.compute_performance(
                    result, data_noise, data_unmasked, data_missing,
                    metadata['input_sizes'],
                    metadata['feature_discrete'],
                    metadata['target_discrete'],
                    metadata['target_numeric']
                )

                total_loss += perf['loss_target'].item()
                if perf['target_count'] > 0:
                    total_accuracy += perf['accuracy_target'].item()
                n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'accuracy': total_accuracy / n_batches if len(metadata['target_discrete']) > 0 else None
        }

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            metadata: Dict,
            n_epochs: int = 100,
            patience: int = 10):
        """
        Train the model with early stopping
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            # Update annealing factor (cosine schedule)
            progress = epoch / n_epochs
            self.annealing_factor = self.final_feature_loss_weight + \
                0.5 * (self.initial_feature_loss_weight - self.final_feature_loss_weight) * \
                (1.0 + np.cos(progress * np.pi))

            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, metadata)

            # Validate
            val_metrics = self.evaluate(val_loader, metadata)

            # Learning rate schedule
            scheduler.step()

            # Print progress
            print(f'\nEpoch {epoch+1}/{n_epochs}')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, '
                  f'Feature Loss: {train_metrics["loss_feature"]:.4f}, '
                  f'Target Loss: {train_metrics["loss_target"]:.4f}')
            if len(metadata['target_discrete']) > 0:
                print(f'  Train - Target Acc: {train_metrics["accuracy_target"]:.2%}')
                print(f'  Val - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.2%}')
            else:
                print(f'  Val - Loss: {val_metrics["loss"]:.4f}')
            print(f'  LR: {scheduler.get_last_lr()[0]:.6f}, Annealing: {self.annealing_factor:.3f}')

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_hopular_model.pt')
                print('  → Model saved!')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping at epoch {epoch+1}')
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_hopular_model.pt'))
        print('\nTraining complete! Best model loaded.')


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                       metadata, batch_size=32):
    """Create PyTorch data loaders"""
    from customhopular import TabularDataset

    # Target indices (last feature is always the target)
    target_indices = metadata['target_discrete'] + metadata['target_numeric']

    train_dataset = TabularDataset(
        X_train, y_train, metadata['input_sizes'], target_indices, mask_prob=0.15
    )
    train_dataset.training = True

    val_dataset = TabularDataset(
        X_val, y_val, metadata['input_sizes'], target_indices, mask_prob=0.0
    )
    val_dataset.training = False

    test_dataset = TabularDataset(
        X_test, y_test, metadata['input_sizes'], target_indices, mask_prob=0.0
    )
    test_dataset.training = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)

    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    import argparse
    import torch
    from customhopular import load_and_preprocess_csv, Hopular

    parser = argparse.ArgumentParser(description="Train Hopular on a CSV dataset")

    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the CSV file"
    )

    parser.add_argument(
        "--target", type=str, required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Test size fraction"
    )

    parser.add_argument(
        "--min_class_samples", type=int, default=2,
        help="Minimum samples per class before filtering"
    )

    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch", type=int, default=32,
        help="Batch size"
    )

    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience"
    )

    args = parser.parse_args()

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = load_and_preprocess_csv(
        csv_path=args.data,
        target_column=args.target,
        test_size=args.test_size,
        min_class_samples=args.min_class_samples
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, metadata, batch_size=args.batch
    )

    # Create model
    model = Hopular(
        input_sizes=metadata['input_sizes'],
        target_discrete=metadata['target_discrete'],
        target_numeric=metadata['target_numeric'],
        feature_discrete=metadata['feature_discrete'],
        memory=metadata['memory'],
        feature_size=32,
        hidden_size=16,
        hidden_size_factor=1.0,
        num_heads=4,
        num_blocks=1,
        scaling_factor=1.0,
        input_dropout=0.2,
        lookup_dropout=0.2,
        output_dropout=0.2,
    )

    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    trainer = HopularTrainer(
        model,
        learning_rate=1e-3,
        weight_decay=0.01,
        initial_feature_loss_weight=1.0,
        final_feature_loss_weight=0.0
    )  
    
    trainer.fit(train_loader, val_loader, metadata, n_epochs=args.epochs, patience=args.patience)

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, metadata)
    print(f'\nTest Results:')
    print(f'  Loss: {test_metrics["loss"]:.4f}')
    if test_metrics['accuracy'] is not None:
        print(f'  Accuracy: {test_metrics["accuracy"]:.2%}')