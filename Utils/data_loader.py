import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_consistent_length, column_or_1d

def load_npz_dataset(dataset_path: str):
    """
    Load and preprocess NPZ format dataset for anomaly detection experiments.
    
    Parameters:
        dataset_path (str): Path to NPZ dataset file
        
    Returns:
        tuple: (X_train, X_test, y_test, dataset_name)
            - X_train (np.ndarray): Normalized training features
            - X_test (np.ndarray): Normalized test features
            - y_test (np.ndarray): 1D test labels (1=anomaly, 0=normal)
            - dataset_name (str): Name of the dataset (without .npz extension)
    """
    # Extract dataset name
    dataset_name = os.path.basename(dataset_path).replace(".npz", "")
    
    # Load raw data
    data = np.load(dataset_path)
    X_train_raw = data['x_train']
    X_test_raw = data['x_test']
    y_test_raw = data['y_test'].flatten()
    
    # Validate data dimensions
    assert X_train_raw.ndim == 2, "Training data must be 2-dimensional (samples × features)"
    assert X_test_raw.ndim == 2, "Test data must be 2-dimensional (samples × features)"
    assert len(X_test_raw) == len(y_test_raw), "Test features and labels must have matching lengths"
    
    # Normalization (train-only fit to avoid data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Ensure labels are 1D and consistent
    y_test = column_or_1d(y_test_raw)
    check_consistent_length(X_test, y_test)
    
    # Log dataset statistics
    print(f"Dataset: {dataset_name}")
    print(f"  - Training samples: {len(X_train)} | Features: {X_train.shape[1]}")
    print(f"  - Test samples: {len(X_test)} | Anomalies: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)")
    
    return X_train, X_test, y_test, dataset_name

def get_dataset_list(dataset_dir: str) -> list:
    """
    Get list of NPZ datasets in target directory.
    
    Parameters:
        dataset_dir (str): Path to directory containing NPZ datasets
        
    Returns:
        list: Full paths to NPZ dataset files
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    dataset_paths = [
        os.path.join(dataset_dir, f) 
        for f in os.listdir(dataset_dir) 
        if f.endswith('.npz')
    ]
    
    if not dataset_paths:
        raise ValueError(f"No NPZ datasets found in: {dataset_dir}")
    
    return dataset_paths