# Hopular: Modern Hopfield Networks for Tabular Data

## Overview

Hopular is a deep learning architecture based on continuous modern Hopfield networks designed specifically for tabular data. It addresses the challenge that traditional deep learning methods often underperform on tabular datasets compared to tree-based methods like gradient boosting. Hopular uses modern Hopfield networks to identify feature-feature, feature-target, and sample-sample dependencies, with each layer having direct access to the original input and the entire training set.

## Key Features

- Modern Hopfield networks for associative memory in tabular data
- Iterative refinement of predictions through multiple blocks
- Support for classification task
- Effective on small to medium-sized tabular datasets
- Memory-augmented architecture that can access training data during inference

## Installation

Make sure you have the following prerequisites installed:

```bash
pip install torch scikit-learn pandas numpy
```

---

## 🌱 PlantAdvisor Dataset Preprocessing

This repository contains a preprocessing script that converts the raw PlantAdvisor Excel dataset (`.xlsx`) into a clean `data.csv` suitable for machine learning training (e.g., Hopular models).

---

### ✨ What the Script Does

- Loads an Excel dataset (header starting on row 2)
- Fills missing values for:
  - `Latitude`
  - `Longitude`
- Automatically drops unwanted columns if present:
  - `Nama Daerah`
  - `Evidence`
- Detects all categorical columns and applies `LabelEncoder`
- Saves the cleaned result into a `.csv` file

---

### 📦 Requirements

Install dependencies:

```bash
pip install pandas scikit-learn
```

---

### 🚀 How to Run

Use the script from the terminal:

```bash
python preprocessing.py --input Dataset_PlantAdvisor.xlsx --output data.csv
```

---

### Arguments
| Argument   | Required | Description                               |
| ---------- | -------- | ----------------------------------------- |
| `--input`  | ✔ Yes    | Path to the input `.xlsx` file            |
| `--output` | ✖ No     | Output CSV filename (default: `data.csv`) |

Example:

```bash
python preprocessing.py --input ./raw/Dataset_PlantAdvisor.xlsx
```

This will create:

```bash
data.csv
```

---

### 📁 Output

- The final output is a cleaned CSV file containing:
- No missing latitude/longitude
- Converted categorical features
- No unused metadata columns
- Ready for machine learning pipelines

---

### 🧠 Script Workflow

- Load .xlsx
- Fill missing coordinates
- Drop unused columns
- Detect and encode categorical fields
- Export to CSV

---

## 🚀 Hopular CSV Training Script

This repository contains a complete training pipeline for the **Hopular** model using tabular data stored in CSV format.

The script includes:

- End-to-end data loading and preprocessing
- Train/validation/test dataloaders
- Full Hopular training loop with:
  - Cosine LR scheduling
  - Feature loss annealing
  - Early stopping
- Model checkpointing (`best_hopular_model.pt`)
- CLI via `argparse`

---

### 📦 Requirements

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn tqdm
```

You also need your customhopular.py file, which must provide:

- Hopular model class
- load_and_preprocess_csv
- TabularDataset
---
### 📁 Input Requirements

Your dataset should be a CSV file, with one column acting as the target label.

The training script will:

- Split into train/validation/test sets
- Encode categorical fields automatically (if enabled in your loader function)
- Prepare Hopular-compatible tensors
---
### 🚀 How to Run

Basic command:
```bash
python train_hopular.py --data data.csv --target Tanaman
```
### Available Arguments
| Argument              | Type  | Required | Description                                       |
| --------------------- | ----- | -------- | ------------------------------------------------- |
| `--data`              | str   | ✔        | Path to the input CSV file                        |
| `--target`            | str   | ✔        | Column name of the prediction target              |
| `--test_size`         | float | ✖        | Size of validation/test split (default: `0.2`)    |
| `--min_class_samples` | int   | ✖        | Minimum samples required per class (default: `2`) |
| `--epochs`            | int   | ✖        | Number of training epochs (default: `100`)        |
| `--batch`             | int   | ✖        | Batch size (default: `32`)                        |
| `--patience`          | int   | ✖        | Early stopping patience (default: `10`)           |

### Example Commands:
Train on dataset with default options
```bash
python train_hopular.py --data Dataset_PlantAdvisor.csv --target Tanaman
```
Train for only 20 epochs
```bash
python train_hopular.py --data data.csv --target Tanaman --epochs 20
```
Use smaller batch size
```bash
python train_hopular.py --data data.csv --target Tanaman --batch 16
```
Custom split sizes
```bash
python train_hopular.py \
  --data data.csv \
  --target Tanaman \
  --test_size 0.3

```

---

### 🧠 Training Outputs

During training, you will see logs like:
```bash
Epoch 1/100
  Train - Loss: 0.8931, Feature Loss: 0.4123, Target Loss: 0.7122
  Train - Target Acc: 52.33%
  Val - Loss: 0.7120, Acc: 55.41%
  LR: 0.001000, Annealing: 1.000
  → Model saved!

```
If early stopping triggers, you will see:
```bash
Early stopping at epoch 24
```

After training completes, the script automatically reloads:
```bash
best_hopular_model.pt
```


---

## 🤖 Hopular Inference / Prediction

This repository includes a complete inference module for making predictions with trained Hopular models.

---

### 📋 Requirements for Inference

After training, you need two files:
- `best_hopular_model.pt` - Trained model weights
- `metadata.pkl` - Preprocessing metadata (scalers, encoders, etc.)

These are automatically saved during training.

---

### 🚀 How to Make Predictions

#### Method 1: Direct API Use
```python
from inference import HopularInference

# Load the trained model and metadata
hopular_inf = HopularInference(
    model_path='best_hopular_model.pt',
    metadata_path='metadata.pkl'
)

# Prepare your input data (must match training format)
import pandas as pd
new_data = pd.DataFrame({
    # Your feature columns here with new values
})

# Make predictions
predictions = hopular_inf.predict(new_data)
print(predictions)
```

#### Method 2: From CSV File
```bash
python inference.py --input_file new_data.csv --output_file predictions.csv
```

#### Method 3: Command Line Interface
```bash
python inference.py \\
  --model_path best_hopular_model.pt \\
  --metadata_path metadata.pkl \\
  --input_file data.csv \\
  --output_file results.csv \\
  --target_column Tanaman  # optional: exclude target column
```

---

### 💡 Inference Arguments
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--model_path` | str | ✖ | Path to model checkpoint (default: `best_hopular_model.pt`) |
| `--metadata_path` | str | ✖ | Path to metadata file (default: `metadata.pkl`) |
| `--input_file` | str | ✔ | Input CSV file for prediction |
| `--output_file` | str | ✖ | Output file for predictions (default: `predictions.csv`) |
| `--target_column` | str | ✖ | Target column to exclude from prediction |
| `--device` | str | ✖ | Device for inference (`cpu` or `cuda`, auto by default) |

---

### 📊 Expected Input Format

Your new data must have the same features and format as the training data:
- Same column names and order
- Same data types
- Categorical values should be from the same set as training (unknown values will be mapped to known ones)

---

### ✅ Output

The inference returns:
- **Classification**: Predicted class labels
- **CSV output**: Input data with additional `prediction` column