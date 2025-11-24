# Fall 2025 ML Course Project

## Dairy Cow Milk Production Prediction

This repository contains the implementation for a Kaggle competition focused on predicting dairy cow milk production using machine learning techniques.

## Project Structure

```
├── data/                          # Dataset directory
│   ├── raw/                       # Raw competition data
│   └── processed/                 # Processed data files
├── notebooks/                     # Python scripts for model pipelines
│   ├── improved_pipeline_v2.py    # Pipeline V2 - Addressing overfitting
│   ├── improved_pipeline_v3.py    # Pipeline V3 - Smart feature engineering
│   ├── improved_pipeline_v4.py    # Pipeline V4 - Fix target encoding leakage
│   ├── improved_pipeline_v5.py    # Pipeline V5 - LightGBM integration
│   ├── improved_pipeline_v6.py    # Pipeline V6 - Aggressive optimization
│   ├── improved_pipeline_v7.py    # Pipeline V7 - Ultra-regularization
│   ├── improved_pipeline_v8.py    # Pipeline V8 - Conservative tuning
│   ├── improved_pipeline_v9.py    # Pipeline V9 - Final optimized model ⭐
│   ├── improved_model.py          # Strategy A+B implementation
│   └── main_analysis.ipynb        # Exploratory data analysis notebook
├── models/                        # Saved model files
├── submissions/                   # Generated submission files
│   └── submission_v9_multiseed_*.csv  # Final submission file
├── Final_Model_V9.ipynb          # Final model notebook
├── Competition Specification.md   # Competition requirements
├── report.md                      # Project report (Markdown)
├── report.pdf                     # Project report (PDF)
├── requirements.txt               # Python dependencies
└── README.md                      # This file

```

## Key Files to Submit

### 1. Final Model Code
- **Primary**: `notebooks/improved_pipeline_v9.py` - Best performing model (CV RMSE: ~4.1211)
- **Alternative**: `Final_Model_V9.ipynb` - Jupyter notebook version
- **Strategy Implementation**: `notebooks/improved_model.py` - Strategy A+B approach

### 2. Submission File
- **Best Result**: `submissions/submission_v9_multiseed_20251123_234237.csv`
- Target performance: Top 20% on leaderboard

### 3. Documentation
- `report.md` / `report.pdf` - Comprehensive project report including:
  - Problem analysis
  - Feature engineering strategies
  - Model selection and tuning
  - Results and evaluation

### 4. Analysis
- `main_analysis.ipynb` - Data exploration and feature analysis
- `Competition Specification.md` - Competition details and requirements

## Model Evolution

The project evolved through multiple iterations:

1. **V1**: Baseline with XGBoost (CV: 4.1271)
2. **V2**: Overfitting reduction (dropped high-cardinality features)
3. **V3**: Smart feature engineering (date features, target encoding)
4. **V4**: Fixed target encoding leakage
5. **V5**: LightGBM integration (CV: ~4.1193)
6. **V6**: 3-way interactions and polynomial features (CV: ~4.1186)
7. **V7**: Ultra-regularization approach (CV: ~4.1209)
8. **V8**: Conservative micro-tuning (CV: ~4.1205)
9. **V9**: Final optimized model with multi-seed ensemble ⭐ (CV: ~4.1211)

## Technical Highlights

### Feature Engineering
- **60+ engineered features** including:
  - Temporal features (cyclical encoding for seasonality)
  - Farm statistical features
  - 2-way and 3-way interaction terms
  - Polynomial features (squared, cubed, log transforms)
  - Domain-specific features (lactation curve, body condition, heat stress)

### Model Architecture
- **Base Models**: XGBoost, LightGBM, ExtraTrees
- **Ensemble Strategy**: Stacking with Ridge meta-learner
- **Validation**: 5-fold cross-validation
- **Optimization**: Hyperparameter tuning for regularization

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn

## Usage

### Run Final Model (V9)
```bash
cd notebooks
python improved_pipeline_v9.py
```

### Generate Predictions
The script will:
1. Load and preprocess data
2. Engineer features
3. Train ensemble model
4. Generate submission file in `submissions/` directory

## Performance Summary

| Version | CV RMSE | Submit RMSE | Gap    | Notes               |
| ------- | ------- | ----------- | ------ | ------------------- |
| V1      | 4.1271  | 4.17495     | 0.048  | Baseline            |
| V5      | 4.1193  | 4.16787     | 0.048  | LightGBM added      |
| V6      | 4.1186  | 4.16585     | 0.047  | 3-way interactions  |
| V8      | 4.1205  | 4.16577     | 0.045  | Conservative tuning |
| V9      | 4.1211  | 4.16438     | 0.043  | Multi-seed ensemble |

## Competition Target

- **Objective**: Predict `Milk_Yield_L` (milk production in liters)
- **Metric**: RMSE (Root Mean Squared Error)
