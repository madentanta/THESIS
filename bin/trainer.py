# hopular_train_classification.py
"""
Classification-only Hopular trainer (target included as last slice).
- Expects preprocessed CSV where features are normalized/encoded and the target column contains
  integer class labels (0..n_classes-1).
- This script will one-hot the target and append it to inputs, mask the target during training,
  train Hopular, evaluate, and save a confusion matrix + metadata.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# import Hopular and TabularDataset from your custom implementation
from customhopular import Hopular, TabularDataset

os.makedirs("output", exist_ok=True)


def create_metadata_from_preprocessed(df: pd.DataFrame, target_column: str):
    """
    Build metadata so Hopular output includes target as the last slice.
    """
    feature_cols = [c for c in df.columns if c != target_column]
    unique_targets = np.unique(df[target_column].values)
    n_classes = int(len(unique_targets))

    # Features are continuous (size=1 each); append target slice size = n_classes
    input_sizes = [1 for _ in feature_cols] + [n_classes]
    target_idx = len(feature_cols)  # index of the target in the slices

    metadata = {
        "feature_names": feature_cols,
        "target_name": target_column,
        "n_classes": n_classes,
        "input_sizes": input_sizes,
        "target_discrete": [target_idx],
        "target_numeric": [],
        "feature_discrete": [],  # for preprocessed, often none
        "task": "classification",
        "target_idx": target_idx,
        "memory": None,  # will set later to training set (features + one-hot target)
    }

    print("Metadata:")
    print(f" - n_features: {len(feature_cols)}")
    print(f" - n_classes: {n_classes}")
    print(f" - input_sizes: {input_sizes}")
    print(f" - target_index: {target_idx}")

    return metadata


def prepare_data_with_target(df: pd.DataFrame, target_column: str, metadata: dict):
    """
    Return X_full (features + one-hot target) and integer class labels (y_int).
    Features are taken as-is (preprocessed).
    """
    feature_cols = metadata["feature_names"]
    n_classes = metadata["n_classes"]

    X_features = df[feature_cols].values.astype(np.float32)
    y_int = df[target_column].values.astype(int)

    # one-hot encode target
    y_onehot = F.one_hot(torch.LongTensor(y_int), num_classes=n_classes).float().numpy()

    # concatenate to form full input (features + one-hot target)
    X_full = np.concatenate([X_features, y_onehot], axis=1).astype(np.float32)

    return X_full, y_int


class TargetMaskingDataset(TabularDataset):
    """
    Dataset that (optionally) masks the target slice with high probability during training
    so Hopular must predict it from context (paper-style).
    """

    def __init__(self, X, y, sizes, target_indices, mask_prob=0.15, target_mask_prob=0.9, force_mask_target=False):
        super().__init__(X, y, sizes, target_indices, mask_prob=mask_prob)
        self.target_mask_prob = target_mask_prob
        self.force_mask_target = force_mask_target

    def __getitem__(self, idx):
        # copied from TabularDataset but with special handling for target masking
        x = self.X[idx]
        y_target = self.y[idx]

        x_masked = x.clone()
        mask = torch.zeros(len(self.sizes), dtype=torch.bool)

        if self.training:
            # target index (first target index)
            target_idx = self.target_indices[0] if len(self.target_indices) > 0 else None
            for i in range(len(self.sizes)):
                start, end = self.boundaries[i], self.boundaries[i + 1]
                if i == target_idx:
                    # high chance to mask target during training
                    if (torch.rand(1) < self.target_mask_prob) or self.force_mask_target:
                        x_masked[start:end] = 0
                        mask[i] = True
                else:
                    # mask other features with mask_prob
                    if torch.rand(1) < self.mask_prob:
                        x_masked[start:end] = 0
                        mask[i] = True

        # missing flags (no explicit missing information here)
        missing = torch.zeros(len(self.sizes), dtype=torch.bool)

        return x_masked, mask, x, missing, torch.tensor(idx, dtype=torch.long)


def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, metadata, batch_size=32, target_mask_prob=0.9):
    target_indices = metadata["target_discrete"] + metadata["target_numeric"]

    train_ds = TargetMaskingDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train),
        metadata["input_sizes"], target_indices,
        mask_prob=0.15, target_mask_prob=target_mask_prob, force_mask_target=False
    )
    train_ds.training = True

    # For validation/test we want the model to predict the masked target too:
    val_ds = TargetMaskingDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val),
        metadata["input_sizes"], target_indices,
        mask_prob=0.0, target_mask_prob=1.0, force_mask_target=True
    )
    # Keep dataset in "training mode" so masking happens in __getitem__.
    # Model will still be eval() during evaluation.
    val_ds.training = True

    test_ds = TargetMaskingDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test),
        metadata["input_sizes"], target_indices,
        mask_prob=0.0, target_mask_prob=1.0, force_mask_target=True
    )
    test_ds.training = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader


class ClassificationHopularTrainer:
    def __init__(self, model, device=None, lr=1e-3, weight_decay=0.0,
                 initial_feature_loss_weight=0.3, final_feature_loss_weight=0.1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_discrete = nn.CrossEntropyLoss()
        self.loss_numeric = nn.MSELoss()
        self.initial_feature_loss_weight = initial_feature_loss_weight
        self.final_feature_loss_weight = final_feature_loss_weight
        self.annealing_factor = initial_feature_loss_weight

    def compute_performance(self, result, data_noise, data_unmasked, data_missing, input_sizes, feature_discrete, target_discrete, target_numeric):
        """
        result: model output (batch, sum(input_sizes))
        data_unmasked: original full values (batch, sum(input_sizes)) including one-hot target at end
        data_noise: mask indicators which features were masked (batch, n_features)
        data_missing: missing indicators (batch, n_features)
        """
        device = result.device
        boundaries = torch.cumsum(torch.tensor([0] + input_sizes, device=device), dim=0)

        loss_feature = torch.tensor(0.0, device=device)
        loss_target = torch.tensor(0.0, device=device)
        acc_feature = 0.0
        acc_target = 0.0
        feature_count = 0
        target_count = 0

        for i in range(len(input_sizes)):
            start = int(boundaries[i].item())
            end = int(boundaries[i + 1].item())

            # which samples had this feature masked AND not missing
            # For proper training, we calculate loss for all samples that had the feature masked,
            # regardless of whether it's a feature or target
            if (i in feature_discrete) or (i in target_discrete):
                # classification: pred logits shape (num_batch, n_classes), tgt one-hot (num_batch, n_classes)
                pred = result[:, start:end]
                tgt = data_unmasked[:, start:end]

                tgt_idx = tgt.argmax(dim=1)

                # Always compute the loss for all samples for this feature/target
                loss = self.loss_discrete(pred, tgt_idx)

                acc = (pred.argmax(dim=1) == tgt_idx).float().mean().item()

                if i in target_discrete:
                    loss_target = loss_target + loss
                    acc_target += acc
                    target_count += 1
                else:
                    loss_feature = loss_feature + loss
                    acc_feature += acc
                    feature_count += 1
            else:
                # regression (continuous)
                # For continuous features, only compute loss on masked samples
                valid = (~data_missing[:, i]) & (data_noise[:, i])
                if not valid.any():
                    continue

                pred = result[valid, start:end]
                tgt = data_unmasked[valid, start:end]

                loss = self.loss_numeric(pred, tgt)
                if i in target_numeric:
                    loss_target = loss_target + loss
                    target_count += 1
                else:
                    loss_feature = loss_feature + loss
                    feature_count += 1

        if feature_count > 0:
            loss_feature = loss_feature / feature_count
            if torch.isnan(loss_feature):
                loss_feature = torch.tensor(0.0, device=device)  # Handle case where no features were masked
        if target_count > 0:
            loss_target = loss_target / target_count
            if torch.isnan(loss_target):
                loss_target = torch.tensor(0.0, device=device)  # Handle case where no targets were masked
            acc_target = acc_target / target_count

        return {
            "loss_feature": loss_feature,
            "loss_target": loss_target,
            "acc_feature": acc_feature,
            "acc_target": acc_target,
            "feature_count": feature_count,
            "target_count": target_count
        }

    def train_epoch(self, train_loader, metadata):
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in tqdm(train_loader, desc="Training"):
            x_masked, data_noise, data_unmasked, data_missing, data_indices = batch
            x_masked = x_masked.to(self.device)
            data_noise = data_noise.to(self.device)
            data_unmasked = data_unmasked.to(self.device)
            data_missing = data_missing.to(self.device)
            data_indices = data_indices.to(self.device)

            # memory mask (prevent self lookup)
            memory_mask = data_indices.view(-1, 1) == self.model.memory_ids.view(1, -1)

            self.optimizer.zero_grad()
            result = self.model(x_masked, memory_mask=memory_mask)

            perf = self.compute_performance(
                result, data_noise, data_unmasked, data_missing,
                metadata["input_sizes"], metadata["feature_discrete"],
                metadata["target_discrete"], metadata["target_numeric"]
            )

            # combine losses - prioritize target loss for better classification
            if perf["target_count"] > 0 and perf["feature_count"] > 0:
                anneal = self.annealing_factor
                # Focus more on target prediction
                loss = (anneal * perf["loss_feature"] + (1.0 - anneal) * perf["loss_target"])
            elif perf["target_count"] > 0:
                loss = perf["loss_target"]  # Pure target prediction loss
            else:
                loss = perf["loss_feature"]

            # only backprop if loss has grad (i.e., some real loss)
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += float(loss.item())
                total_batches += 1
            else:
                # nothing to optimize this batch (no masked features/targets)
                continue

        return {"loss": total_loss / total_batches if total_batches > 0 else 0.0}

    def evaluate(self, loader, metadata, collect_predictions=False):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                x_masked, data_noise, data_unmasked, data_missing, data_indices = batch
                x_masked = x_masked.to(self.device)
                data_noise = data_noise.to(self.device)
                data_unmasked = data_unmasked.to(self.device)
                data_missing = data_missing.to(self.device)

                result = self.model(x_masked, memory_mask=None)

                perf = self.compute_performance(
                    result, data_noise, data_unmasked, data_missing,
                    metadata["input_sizes"], metadata["feature_discrete"],
                    metadata["target_discrete"], metadata["target_numeric"]
                )

                # pick evaluation loss (prioritize target)
                if perf["target_count"] > 0:
                    eval_loss = perf["loss_target"]
                else:
                    eval_loss = perf["loss_feature"]

                total_loss += float(eval_loss.item()) if isinstance(eval_loss, torch.Tensor) else float(eval_loss)
                n_batches += 1

                # collect predicted classes for the target
                if collect_predictions and len(metadata["target_discrete"]) > 0:
                    tgt_idx = metadata["target_discrete"][0]
                    boundaries = torch.cumsum(torch.tensor([0] + metadata["input_sizes"], device=result.device), dim=0)
                    start = int(boundaries[tgt_idx].item())
                    end = int(boundaries[tgt_idx + 1].item())

                    pred_logits = result[:, start:end]  # shape (batch, n_classes)
                    pred_classes = pred_logits.argmax(dim=1).cpu().numpy()

                    true_classes = data_unmasked[:, start:end].argmax(dim=1).cpu().numpy()

                    all_preds.extend(pred_classes.tolist())
                    all_targets.extend(true_classes.tolist())

        return {
            "loss": total_loss / n_batches if n_batches > 0 else 0.0,
            "predictions": np.array(all_preds) if collect_predictions else None,
            "targets": np.array(all_targets) if collect_predictions else None
        }


def plot_confusion_matrix(y_true, y_pred, save_path="output/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")


def main(args):
    # load df
    df = pd.read_csv(args.data)
    print("Loaded:", df.shape, "columns:", df.columns.tolist())
    print("Target value counts:")
    print(df[args.target].value_counts())

    # build metadata
    metadata = create_metadata_from_preprocessed(df, args.target)

    # split indexes
    idx = np.arange(len(df))

    # 1) Test split = 25%
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=0.25,
        random_state=42,
        stratify=df[args.target] if args.stratify else None
    )

    # 2) Validation split = 10% total → 10/75 = 0.1333333 of train_val
    val_relative = 0.10 / 0.75  # = 0.1333333

    tr_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=42,
        stratify=df.iloc[train_val_idx][args.target] if args.stratify else None
    )

    print(f"Split sizes: train={len(tr_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    # prepare X_full and y_int
    X_full_train, y_train_int = prepare_data_with_target(df.iloc[tr_idx], args.target, metadata)
    X_full_val, y_val_int = prepare_data_with_target(df.iloc[val_idx], args.target, metadata)
    X_full_test, y_test_int = prepare_data_with_target(df.iloc[test_idx], args.target, metadata)

    print("Shapes (train/val/test):", X_full_train.shape, X_full_val.shape, X_full_test.shape)

    # set memory to training full matrix (features + one-hot target)
    metadata["memory"] = torch.FloatTensor(X_full_train)
    print("Memory shape:", metadata["memory"].shape)

    # create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_full_train, X_full_val, X_full_test,
        y_train_int, y_val_int, y_test_int,
        metadata, batch_size=args.batch, target_mask_prob=args.target_mask_prob
    )

    # build model
    model = Hopular(
        input_sizes=metadata["input_sizes"],
        target_discrete=metadata["target_discrete"],
        target_numeric=metadata["target_numeric"],
        feature_discrete=metadata["feature_discrete"],
        memory=metadata["memory"],
        feature_size=32,
        hidden_size=32,
        hidden_size_factor=1.0,
        num_heads=4,
        num_blocks=args.num_blocks,
        scaling_factor=1.0,
        input_dropout=0.1,
        lookup_dropout=0.1,
        output_dropout=0.1,
        memory_ratio=1.0
    )

    trainer = ClassificationHopularTrainer(model, lr=args.lr, weight_decay=args.weight_decay,
                                           initial_feature_loss_weight=args.init_feat_w, final_feature_loss_weight=args.final_feat_w)

    # train loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = "output/best_hopular_model.pt"

    for epoch in range(args.epochs):
        # annealing schedule
        progress = epoch / max(1, args.epochs)
        trainer.annealing_factor = trainer.final_feature_loss_weight + 0.5 * (trainer.initial_feature_loss_weight - trainer.final_feature_loss_weight) * (1.0 + np.cos(progress * np.pi))

        train_res = trainer.train_epoch(train_loader, metadata)
        val_res = trainer.evaluate(val_loader, metadata, collect_predictions=False)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_res['loss']:.6f} | Val Loss: {val_res['loss']:.6f} | Anneal: {trainer.annealing_factor:.3f}")

        # save best model by validation loss
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    # Ensure a model exists to load. If not, save current model.
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)
        print("No best model found - saved final model to output/best_hopular_model.pt")

    # load best model and evaluate on test set collecting predictions
    model.load_state_dict(torch.load(best_path, map_location=trainer.device))
    eval_res = trainer.evaluate(test_loader, metadata, collect_predictions=True)
    print("Test loss:", eval_res["loss"])

    preds = eval_res["predictions"]
    targets = eval_res["targets"]

    if preds is None or len(preds) == 0:
        print("No predictions collected. Check dataset/masking configuration.")
    else:
        print("Accuracy:", accuracy_score(targets, preds))
        print("Classification Report:\n", classification_report(targets, preds))
        plot_confusion_matrix(targets, preds, save_path="output/confusion_matrix.png")

    # save metadata
    import pickle
    with open("output/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("Saved metadata to output/metadata.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Preprocessed CSV (features normalized; target integer labels)")
    parser.add_argument("--target", required=True, help="Target column name (integer labels 0..n-1)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--target_mask_prob", type=float, default=0.9)
    parser.add_argument("--init_feat_w", type=float, default=0.3)
    parser.add_argument("--final_feat_w", type=float, default=0.1)
    parser.add_argument("--stratify", action="store_true", help="Use stratified splits")
    args = parser.parse_args()
    main(args)