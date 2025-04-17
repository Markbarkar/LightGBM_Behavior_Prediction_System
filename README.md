# Advanced Behavior Prediction System

## Overview
This project implements an advanced behavior prediction system using multiple machine learning models (XGBoost, CatBoost, LightGBM) with ensemble learning. The system is designed to predict user behavior types based on historical data, with robust feature engineering and comprehensive model evaluation.

## Features
- **Multi-Model Integration**: Combines XGBoost, CatBoost, and LightGBM with voting ensemble
- **Advanced Feature Engineering**: 
  - Time-based features (hour, day, month, day of week)
  - User behavior statistics
  - Item behavior statistics
  - Shop behavior statistics
  - Interaction features
  - Time window features
- **GPU Acceleration**: Supports multi-GPU training (up to 2 GPUs)
- **Checkpoint & Resume**: Automatic model checkpointing and training resumption
- **Comprehensive Visualization**:
  - Class distribution plots
  - Feature importance analysis
  - Confusion matrices
  - Learning curves
- **Robust Error Handling**: Graceful handling of missing data and feature calculation errors

## Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Required Python packages:
  ```
  pandas
  numpy
  xgboost
  catboost
  lightgbm
  scikit-learn
  matplotlib
  seaborn
  joblib
  ```

## Project Structure
```
.
├── data/
│   └── processed/
│       ├── train.csv
│       └── val.csv
├── models/
│   └── checkpoints/
│       ├── xgb_checkpoint.json
│       ├── cb_checkpoint.cbm
│       └── lgb_checkpoint.txt
├── plots/
│   ├── class_distribution.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── learning_curves.png
├── advanced_behavior_prediction.py
└── README.md
```

## Usage
1. **Data Preparation**:
   - Place your training data in `data/processed/train.csv`
   - Place your validation data in `data/processed/val.csv`

2. **Training**:
   ```bash
   python advanced_behavior_prediction.py
   ```
   The script will:
   - Load and preprocess the data
   - Perform feature engineering
   - Train multiple models with GPU acceleration
   - Save checkpoints every 10 iterations
   - Generate evaluation plots
   - Save final models

3. **Model Checkpoints**:
   - Checkpoints are saved in `models/checkpoints/`
   - Training can be resumed from the last checkpoint if interrupted

## Model Details
- **XGBoost**:
  - GPU-accelerated training
  - Multi-class classification
  - Early stopping with checkpointing

- **CatBoost**:
  - GPU support
  - Categorical feature handling
  - Automatic snapshot saving

- **LightGBM**:
  - GPU acceleration
  - Feature importance tracking
  - Model checkpointing

- **Voting Classifier**:
  - Soft voting ensemble
  - Parallel prediction with all CPU cores

## Performance Features
- **GPU Acceleration**:
  - Supports up to 2 GPUs
  - Automatic GPU detection and utilization
  - Optimized for CUDA-enabled environments

- **Memory Efficiency**:
  - Efficient data loading
  - Optimized feature calculation
  - Memory-friendly batch processing

- **Training Optimization**:
  - Early stopping
  - Automatic checkpointing
  - Progress monitoring

## Output
- **Models**: Saved in `models/` directory
- **Plots**: Generated in `plots/` directory
  - Class distribution
  - Feature importance
  - Confusion matrices
  - Learning curves

## Notes
- The system automatically handles missing data and feature calculation errors
- All visualizations are in English to avoid display issues
- Training progress is displayed in real-time
- Models are saved after each training session