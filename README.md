<p align="center"><a href="https://laravel.com" target="_blank"><img src="https://raw.githubusercontent.com/laravel/art/master/logo-lockup/5%20SVG/2%20CMYK/1%20Full%20Color/laravel-logolockup-cmyk-red.svg" width="400" alt="Laravel Logo"></a></p>

<p align="center">
<a href="https://github.com/laravel/framework/actions"><img src="https://github.com/laravel/framework/workflows/tests/badge.svg" alt="Build Status"></a>
<a href="https://packagist.org/packages/laravel/framework"><img src="https://img.shields.io/packagist/dt/laravel/framework" alt="Total Downloads"></a>
<a href="https://packagist.org/packages/laravel/framework"><img src="https://img.shields.io/packagist/v/laravel/framework" alt="Latest Stable Version"></a>
<a href="https://packagist.org/packages/laravel/framework"><img src="https://img.shields.io/packagist/l/laravel/framework" alt="License"></a>
</p>

## About Laravel

Laravel is a web application framework with expressive, elegant syntax. We believe development must be an enjoyable and creative experience to be truly fulfilling. Laravel takes the pain out of development by easing common tasks used in many web projects, such as:

- [Simple, fast routing engine](https://laravel.com/docs/routing).
- [Powerful dependency injection container](https://laravel.com/docs/container).
- Multiple back-ends for [session](https://laravel.com/docs/session) and [cache](https://laravel.com/docs/cache) storage.
- Expressive, intuitive [database ORM](https://laravel.com/docs/eloquent).
- Database agnostic [schema migrations](https://laravel.com/docs/migrations).
- [Robust background job processing](https://laravel.com/docs/queues).
- [Real-time event broadcasting](https://laravel.com/docs/broadcasting).

Laravel is accessible, powerful, and provides tools required for large, robust applications.

## Learning Laravel

Laravel has the most extensive and thorough [documentation](https://laravel.com/docs) and video tutorial library of all modern web application frameworks, making it a breeze to get started with the framework. You can also check out [Laravel Learn](https://laravel.com/learn), where you will be guided through building a modern Laravel application.

If you don't feel like reading, [Laracasts](https://laracasts.com) can help. Laracasts contains thousands of video tutorials on a range of topics including Laravel, modern PHP, unit testing, and JavaScript. Boost your skills by digging into our comprehensive video library.

## Laravel Sponsors

We would like to extend our thanks to the following sponsors for funding Laravel development. If you are interested in becoming a sponsor, please visit the [Laravel Partners program](https://partners.laravel.com).

### Premium Partners

- **[Vehikl](https://vehikl.com)**
- **[Tighten Co.](https://tighten.co)**
- **[Kirschbaum Development Group](https://kirschbaumdevelopment.com)**
- **[64 Robots](https://64robots.com)**
- **[Curotec](https://www.curotec.com/services/technologies/laravel)**
- **[DevSquad](https://devsquad.com/hire-laravel-developers)**
- **[Redberry](https://redberry.international/laravel-development)**
- **[Active Logic](https://activelogic.com)**

## Contributing

Thank you for considering contributing to the Laravel framework! The contribution guide can be found in the [Laravel documentation](https://laravel.com/docs/contributions).

## Code of Conduct

In order to ensure that the Laravel community is welcoming to all, please review and abide by the [Code of Conduct](https://laravel.com/docs/contributions#code-of-conduct).

## Security Vulnerabilities

If you discover a security vulnerability within Laravel, please send an e-mail to Taylor Otwell via [taylor@laravel.com](mailto:taylor@laravel.com). All security vulnerabilities will be promptly addressed.

## License

The Laravel framework is open-sourced software licensed under the [MIT license](https://opensource.org/licenses/MIT).

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
pip install -r requirements.txt
---

## 🗄️ Fetching Data from Supabase

This repository includes utilities to fetch data directly from a Supabase database and convert it to CSV format for use in the Hopular pipeline.

---

### 🔌 Supabase Connection Setup

1. Get your Supabase credentials:
   - Project URL (e.g., `https://xxxxx.supabase.co`)
   - API key (either anon key or service role key)

2. Install the Supabase client:
```bash
pip install supabase
```

3. Ensure your Supabase table is accessible (public schema and RLS policies permitting)

### 🚀 How to Fetch Data from Supabase

#### Command Line Usage:
```bash
python fetch_from_supabase.py --url YOUR_SUPABASE_URL --key YOUR_SUPABASE_KEY --table TABLE_NAME
```

#### Arguments:
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--url` | str | ✅ | Your Supabase project URL |
| `--key` | str | ✅ | Your Supabase API key (anon or service role) |
| `--table` | str | ✅ | Name of the table to fetch |
| `--output` | str | ❌ | Output CSV filename (default: `supabase_data.csv`) |
| `--filters` | str | ❌ | Filters in format: `col1.eq.value,col2.gt.100` |
| `--limit` | int | ❌ | Row limit |
| `--preview` | flag | ❌ | Print first few rows |

#### Example Commands:

Fetch entire table:
```bash
python fetch_from_supabase.py --url https://myproject.supabase.co --key my_api_key --table my_table
```

With filters and limits:
```bash
python fetch_from_supabase.py \
  --url https://myproject.supabase.co \
  --key my_api_key \
  --table plants \
  --filters "status.eq.active,created_at.gte.2023-01-01" \
  --limit 1000 \
  --output plant_data.csv
```

Fetch and preview:
```bash
python fetch_from_supabase.py --url https://myproject.supabase.co --key my_api_key --table my_table --preview
```

### 🔄 Using the Fetched Data

Once you have fetched your data as CSV, you can use it with the existing Hopular pipeline:

1. Train a model:
```bash
python trainer.py --data supabase_data.csv --target TARGET_COLUMN
```

2. Use in your preprocessing pipeline:
```bash
python preprocessing.py --input supabase_data.csv --output processed_data.csv
```

### ⚠️ Important Notes

- Make sure your Supabase table structure is compatible with your Hopular model (same features/columns)
- The Supabase API key should have appropriate permissions to read the table
- Consider data size limitations when fetching large tables
- Enable Row Level Security (RLS) in Supabase appropriately for your use case

---

## 🚀 Automated Pipeline: run_train

This repository includes a complete automated pipeline that combines fetching from Supabase, preprocessing, and training in a single command using subprocess calls.

### 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### 🚀 How to Use run_train

#### Command Line Usage:
```bash
python run_train.py --url YOUR_SUPABASE_URL --key YOUR_SUPABASE_KEY --table TABLE_NAME --target TARGET_COLUMN
```

#### Arguments:
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--url` | str | ✅ | Your Supabase project URL |
| `--key` | str | ✅ | Your Supabase API key (anon or service role) |
| `--table` | str | ✅ | Name of the Supabase table to fetch |
| `--target` | str | ✅ | Target column name for training |
| `--supabase-csv` | str | ❌ | Raw Supabase data CSV filename (default: `supabase_data.csv`) |
| `--output-csv` | str | ❌ | Processed data CSV filename (default: `processed_data.csv`) |
| `--filters` | str | ❌ | Filters in format: `col1.eq.value,col2.gt.100` |
| `--limit` | int | ❌ | Row limit when fetching from Supabase |
| `--epochs` | int | ❌ | Number of training epochs (default: 100) |
| `--batch-size` | int | ❌ | Batch size for training (default: 32) |
| `--patience` | int | ❌ | Early stopping patience (default: 10) |

#### Example Commands:

Complete pipeline with defaults:
```bash
python run_train.py --url https://myproject.supabase.co --key my_api_key --table my_table --target target_column
```

With filters and custom training parameters:
```bash
python run_train.py \
  --url https://myproject.supabase.co \
  --key my_api_key \
  --table plants \
  --target Tanaman \
  --filters "status.eq.active,created_at.gte.2023-01-01" \
  --limit 1000 \
  --epochs 50 \
  --batch-size 16
```

### 🔄 Pipeline Steps

The `run_train.py` script automatically executes these steps:

1. **Fetch from Supabase**: Calls `fetch_from_supabase.py` to download data
2. **Preprocess**: Calls `preprocessing.py` to clean and prepare the data
3. **Train**: Calls `trainer.py` to train the Hopular model

### 📁 Output Files

When the pipeline completes successfully, you'll have these files:
- `supabase_data.csv` (raw data from Supabase)
- `processed_data.csv` (cleaned and preprocessed data)
- `best_hopular_model.pt` (trained model)
- `metadata.pkl` (preprocessing metadata for inference)

### ✅ Success Indicators

A successful run will show:
- ✅ Fetching from Supabase completed
- ✅ Preprocessing completed
- ✅ Training completed
- ✅ All four output files generated

---

## 🌐 FastAPI for Hopular Pipeline

This repository includes a FastAPI application that exposes both the run_train pipeline and inference as an HTTP API, allowing you to trigger complete workflows or make predictions via API calls.

---

### 🚀 How to Start the API Server

#### Install dependencies:
```bash
pip install -r requirements.txt
```

#### Start the server:
```bash
python api.py
```

By default, the API will be available at `http://localhost:8000`.

#### Or with uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 📡 API Endpoints

#### Root endpoint:
- `GET /` - Check if the API is running

#### Run Train Pipeline:
- `POST /run_train` - Start the complete pipeline (async)
- `GET /run_train/{job_id}` - Check the status of a pipeline job

#### Inference:
- `POST /inference` - Run inference on input data using trained model

### 🛠️ Run Train Pipeline API Usage

#### Start a pipeline job:
```bash
curl -X POST "http://localhost:8000/run_train" \
  -H "Content-Type: application/json" \
  -d '{
    "supabase_url": "https://your-project.supabase.co",
    "supabase_key": "your_supabase_key",
    "table_name": "your_table",
    "target_column": "target_column",
    "epochs": 50,
    "batch_size": 16
  }'
```

The API will return a job ID that you can use to check the status of the pipeline.

#### Check pipeline status:
```bash
curl -X GET "http://localhost:8000/run_train/JOB_ID"
```

Example response:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "completed",
  "message": "Pipeline completed successfully!",
  "progress": "Training the Hopular model",
  "created_at": "2023-10-01T10:00:00",
  "completed_at": "2023-10-01T10:15:00",
  "data": "supabase_data.csv",
  "output_csv": "processed_data.csv",
  "model_path": "best_hopular_model.pt",
  "metadata_path": "metadata.pkl"
}
```

#### Run Train Pipeline Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `supabase_url` | string | ✅ | Your Supabase project URL |
| `supabase_key` | string | ✅ | Your Supabase API key (anon or service role) |
| `table_name` | string | ✅ | Name of the Supabase table to fetch |
| `target_column` | string | ✅ | Target column name for training |
| `data` | string | ❌ | Raw Supabase data CSV filename (default: `supabase_data.csv`) |
| `output_csv` | string | ❌ | Processed data CSV filename (default: `processed_data.csv`) |
| `filters` | string | ❌ | Filters in format: `col1.eq.value,col2.gt.100` |
| `limit` | integer | ❌ | Row limit when fetching from Supabase |
| `epochs` | integer | ❌ | Number of training epochs (default: 100) |
| `batch_size` | integer | ❌ | Batch size for training (default: 32) |
| `patience` | integer | ❌ | Early stopping patience (default: 10) |
| `test_size` | number | ❌ | Test size fraction (default: 0.2) |
| `min_class_samples` | integer | ❌ | Minimum samples per class (default: 2) |

#### Run Train Pipeline Status Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique identifier for the job |
| `status` | string | Job status (`pending`, `running`, `completed`, `failed`) |
| `message` | string | Status message |
| `progress` | string | Current step being executed |
| `created_at` | datetime | When the job was created |
| `completed_at` | datetime | When the job completed (if completed) |
| `data` | string | Path to the Supabase CSV file |
| `output_csv` | string | Path to the processed CSV file |
| `model_path` | string | Path to the trained model file |
| `metadata_path` | string | Path to the metadata file |

### 🧠 Inference API Usage

#### Run inference on input data:
```bash
curl -X POST "http://localhost:8000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "best_hopular_model.pt",
    "metadata_path": "metadata.pkl",
    "target_column": "target_column",
    "input_data": [
      {
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": "category_a"
      },
      {
        "feature1": 1.5,
        "feature2": 2.5,
        "feature3": "category_b"
      }
    ]
  }'
```

#### Example inference response:
```json
{
  "predictions": ["class_a", "class_b"],
  "input_count": 2,
  "model_path": "best_hopular_model.pt",
  "metadata_path": "metadata.pkl",
  "processed_at": "2023-10-01T10:30:00"
}
```

#### Inference Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | string | ❌ | Path to the trained model file (default: `best_hopular_model.pt`) |
| `metadata_path` | string | ❌ | Path to the metadata file (default: `metadata.pkl`) |
| `target_column` | string | ❌ | Target column to exclude from prediction (if exists in input data) |
| `input_data` | array of objects | ✅ | Array of records to predict (each record is an object with feature-value pairs) |

#### Inference Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | array | Array of prediction results |
| `input_count` | integer | Number of records processed |
| `model_path` | string | Path to the model used for inference |
| `metadata_path` | string | Path to the metadata used for inference |
| `processed_at` | datetime | When the inference was processed |

### 🌱 PlantAdvisor Dataset Preprocessing
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
