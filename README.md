 HEAD
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









 326f9298728ebec969c3cdd6938e2fdf9960c878
