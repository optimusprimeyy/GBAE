# ============================================================================
# Data Loader Module
# Handles data loading, preprocessing, and batch generation
# ============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path


class AnomalyDataset(Dataset):
    """
    Custom Dataset for anomaly detection.
    Loads and preprocesses data from .npz files.
    """

    def __init__(self, data, labels=None, normalize=True):
        """
        Args:
            data: numpy array of shape (N, D) where N is sample count, D is dimension
            labels: numpy array of shape (N,) for anomaly labels (1 for anomaly, 0 for normal)
            normalize: whether to normalize the data
        """
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64) if labels is not None else None
        
        if normalize:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx])
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return sample, label
        return sample


class DataLoader_GBAE:
    """
    Data loading and preprocessing wrapper for GBAE.
    Handles .npz file loading, train/test splits, and DataLoader creation.
    """

    def __init__(self, config):
        """
        Args:
            config: dict containing data configuration
        """
        self.config = config
        self.data_dir = config.get('data_dir', 'data/raw/')
        self.processed_dir = config.get('processed_data_dir', 'data/processed/')
        self.normalize = config.get('normalize', True)
        self.random_seed = config.get('random_seed', 42)
        
        # Create processed directory if not exists
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

    def load_npz(self, filename):
        """
        Load data from .npz file.
        
        Args:
            filename: name of the .npz file (without path)
        
        Returns:
            data: numpy array of shape (N, D)
            labels: numpy array of shape (N,) or None if not in file
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = np.load(file_path)
        
        # Try common key names for data and labels
        if 'X' in data:
            X = data['X']
        elif 'data' in data:
            X = data['data']
        else:
            # Use first available array
            X = data[list(data.files)[0]]
        
        # Try to get labels if available
        y = None
        if 'y' in data:
            y = data['y']
        elif 'labels' in data:
            y = data['labels']
        
        return np.asarray(X, dtype=np.float32), y

    def create_train_test_split(self, X, y=None, train_split=0.8):
        """
        Split data into train and test sets.
        
        Args:
            X: feature array
            y: label array (optional)
            train_split: proportion of training data
        
        Returns:
            X_train, X_test, y_train (optional), y_test (optional)
        """
        np.random.seed(self.random_seed)
        n_samples = len(X)
        n_train = int(n_samples * train_split)
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        results = [X_train, X_test]
        
        if y is not None:
            y_train = y[train_idx]
            y_test = y[test_idx]
            results.extend([y_train, y_test])
        
        return tuple(results) if len(results) > 2 else results

    def create_dataloader(self, X, y=None, batch_size=32, shuffle=True, num_workers=0):
        """
        Create PyTorch DataLoader from data.
        
        Args:
            X: feature array
            y: label array (optional)
            batch_size: batch size for dataloader
            shuffle: whether to shuffle data
            num_workers: number of workers for data loading
        
        Returns:
            DataLoader object
        """
        dataset = AnomalyDataset(X, y, normalize=self.normalize)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def load_and_prepare(self, filename, batch_size=32, train_split=0.8):
        """
        Load .npz file, split data, and create dataloaders.
        
        Args:
            filename: name of the .npz file
            batch_size: batch size for dataloaders
            train_split: proportion of training data
        
        Returns:
            train_loader, test_loader, input_dim
        """
        # Load data
        X, y = self.load_npz(filename)
        input_dim = X.shape[1]
        
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = self.create_train_test_split(
                X, y, train_split
            )
        else:
            X_train, X_test = self.create_train_test_split(X, None, train_split)
            y_train, y_test = None, None
        
        # Create dataloaders
        train_loader = self.create_dataloader(
            X_train, y_train, batch_size=batch_size, shuffle=True
        )
        test_loader = self.create_dataloader(
            X_test, y_test, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader, input_dim
