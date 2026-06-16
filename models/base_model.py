import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import k_means

# -------------------------- Granular Computing Module --------------------------
class GranularBall:
    """
    Granular Ball (GB) class for representing data granules with geometric properties.
    
    A Granular Ball is defined by:
    - Data points within the granule
    - Geometric center (mean of feature vectors)
    - Radius (maximum Euclidean distance from center to any data point)
    - Unique index identifier
    """
    def __init__(self, data: np.ndarray, index: int):
        """
        Initialize Granular Ball from data points.
        
        Parameters:
            data (np.ndarray): Data matrix (last column = sample index, no label column)
            index (int): Unique identifier for the granular ball
        """
        self.data = data[:, :-1]  # Extract feature matrix (exclude index column)
        self.index = index
        self.center = self._compute_geometric_center()  # GB center (mean vector)
        self.radius = self._compute_radius()  # GB radius (max distance to center)
        self.score = 0  # Placeholder for anomaly score (unused in base implementation)

    def _compute_geometric_center(self) -> np.ndarray:
        """Compute geometric center (mean) of the granular ball."""
        return np.mean(self.data, axis=0)

    def _compute_radius(self) -> float:
        """
        Compute radius of the granular ball (maximum Euclidean distance from center to data points).
        
        Returns:
            float: Radius (1e-6 if single sample to avoid zero radius)
        """
        if len(self.data) == 1:
            return 1e-6
        
        # Euclidean distance from center to all data points
        distances = np.sqrt(np.sum((self.data - self.center) ** 2, axis=1))
        return np.max(distances)

def calculate_statistical_distance(gb_data: np.ndarray) -> float:
    """
    Calculate Statistical Distance (SD) for a granular ball:
    Sum of Euclidean distances from all points to the GB center.
    
    Parameters:
        gb_data (np.ndarray): Granular ball data matrix (last column = sample index)
        
    Returns:
        float: Total statistical distance of the granular ball
    """
    feature_data = gb_data[:, :-1]
    center = np.mean(feature_data, axis=0)
    return np.sum(np.sqrt(np.sum((feature_data - center) ** 2, axis=1)))

def split_granular_ball(gb_data: np.ndarray) -> tuple:
    """
    Split a granular ball into two sub-balls using 2-means clustering.
    
    Parameters:
        gb_data (np.ndarray): Granular ball data matrix (last column = sample index)
        
    Returns:
        tuple: (sub_ball_1, sub_ball_2) - Two sub-granular balls
    """
    feature_data = gb_data[:, :-1]
    # Perform 2-means clustering
    cluster_labels = k_means(X=feature_data, init='k-means++', n_clusters=2)[1]
    # Split data based on cluster labels
    sub_ball_1 = gb_data[cluster_labels == 0, :]
    sub_ball_2 = gb_data[cluster_labels == 1, :]
    return sub_ball_1, sub_ball_2

def divide_granular_balls(gb_list: list, sample_threshold: int) -> list:
    """
    Divide granular balls based on SD minimization criterion:
    Split a GB if sum of SD of sub-balls < SD of original GB (and sample count ≥ threshold).
    
    Parameters:
        gb_list (list): List of granular balls (each as np.ndarray)
        sample_threshold (int): Minimum sample count for splitting
        
    Returns:
        list: Updated list of granular balls after division
    """
    new_gb_list = []
    for gb in gb_list:
        if gb.shape[0] >= max(sample_threshold, 2):
            # Split GB into two sub-balls
            sub_ball_1, sub_ball_2 = split_granular_ball(gb)
            
            # Handle edge cases (all samples in one sub-ball)
            if len(sub_ball_1) == 0:
                new_gb_list.append(sub_ball_2)
                continue
            if len(sub_ball_2) == 0:
                new_gb_list.append(sub_ball_1)
                continue
            
            # Calculate SD for original and sub-balls
            sd_original = calculate_statistical_distance(gb)
            sd_sub_1 = calculate_statistical_distance(sub_ball_1)
            sd_sub_2 = calculate_statistical_distance(sub_ball_2)
            sd_sub_total = sd_sub_1 + sd_sub_2
            
            # Split if SD is reduced (SD_sub < SD_original)
            if sd_sub_total < sd_original:
                new_gb_list.extend([sub_ball_1, sub_ball_2])
            else:
                new_gb_list.append(gb)
        else:
            new_gb_list.append(gb)
    return new_gb_list

def construct_granular_balls(data: np.ndarray, delta: float = 0.5) -> list:
    """
    Construct granular balls from raw data using divisive clustering with SD criterion.
    
    Parameters:
        data (np.ndarray): Raw feature matrix (samples × features)
        delta (float): Threshold parameter (unused in base implementation, for extension)
        
    Returns:
        list: Final list of granular balls (each as np.ndarray with sample index column)
    """
    # Set sample threshold (sqrt of total samples)
    sample_threshold = int(np.sqrt(len(data)))
    print(f"Granular Ball Construction: Sample threshold = {sample_threshold}")
    
    # Add sample index column to track original samples
    sample_indices = np.arange(data.shape[0]).reshape(-1, 1)
    data_with_indices = np.hstack([data, sample_indices])
    data_with_indices[:, -1] = data_with_indices[:, -1].astype(int)
    
    # Initialize with entire dataset as one granular ball
    current_gb_list = [data_with_indices]
    
    # Iterative division until no more splits
    while True:
        num_gb_before = len(current_gb_list)
        current_gb_list = divide_granular_balls(current_gb_list, sample_threshold)
        num_gb_after = len(current_gb_list)
        
        # Terminate if no new granular balls are created
        if num_gb_after == num_gb_before:
            break
    
    return current_gb_list

def create_granular_ball_instances(gb_data_list: list) -> tuple:
    """
    Create GranularBall class instances from raw granular ball data.
    
    Parameters:
        gb_data_list (list): List of raw granular ball data (np.ndarray)
        
    Returns:
        tuple: (gb_instances, center_matrix)
            - gb_instances (list): List of GranularBall objects
            - center_matrix (np.ndarray): Matrix of GB centers (n_gb × n_features)
    """
    gb_instances = []
    center_matrix = []
    
    for idx, gb_data in enumerate(gb_data_list):
        gb = GranularBall(gb_data, idx)
        gb_instances.append(gb)
        center_matrix.append(gb.center)
    
    return gb_instances, np.array(center_matrix)

# -------------------------- Autoencoder Module --------------------------
class CenterOnlyAutoencoder(nn.Module):
    """
    Lightweight Autoencoder (AE) designed for training on Granular Ball centers only.
    Architecture:
    - Encoder: 2 linear layers with LeakyReLU activation and dropout
    - Decoder: 2 linear layers with LeakyReLU activation and dropout
    """
    def __init__(self, input_dim: int, hidden_dim: int = None, dropout_rate: float = 0.3):
        """
        Initialize Center-Only Autoencoder.
        
        Parameters:
            input_dim (int): Dimensionality of input features (GB centers)
            hidden_dim (int): Dimensionality of latent space (default: max(2, input_dim//4))
            dropout_rate (float): Dropout probability (0-1) for regularization
        """
        super(CenterOnlyAutoencoder, self).__init__()
        
        # Set default hidden dimension if not specified
        if hidden_dim is None:
            hidden_dim = max(2, input_dim // 4)
        
        # Encoder: Input → Latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dim // 2, hidden_dim)
        )
        
        # Decoder: Latent space → Reconstructed input
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the autoencoder.
        
        Parameters:
            x (torch.Tensor): Input tensor (batch_size × input_dim)
            
        Returns:
            tuple: (reconstructed_x, latent_z)
                - reconstructed_x: Reconstructed input (batch_size × input_dim)
                - latent_z: Latent representation (batch_size × hidden_dim)
        """
        latent_z = self.encoder(x)
        reconstructed_x = self.decoder(latent_z)
        return reconstructed_x, latent_z

# -------------------------- Anomaly Scoring Module --------------------------
def robust_normalization(x: np.ndarray) -> np.ndarray:
    """
    Robust normalization using median and Interquartile Range (IQR) to mitigate outlier effects.
    
    Parameters:
        x (np.ndarray): Input array to normalize
        
    Returns:
        np.ndarray: Normalized array (zero median, unit IQR)
    """
    median = np.median(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    # Avoid division by zero (add small epsilon)
    return (x - median) / (iqr + 1e-10)

def compute_anomaly_scores(
    samples: np.ndarray, 
    model: nn.Module, 
    gb_instances: list, 
    device: torch.device
    ) -> tuple:
    """
    Compute anomaly scores combining reconstruction error and latent space consistency error.
    
    Anomaly Score Components:
    1. Score_R (Reconstruction Error): L2 loss between input and reconstructed samples
    2. Score_L (Latent Consistency Error): Absolute difference between:
       - Euclidean distance in original space (sample to nearest GB center)
       - Euclidean distance in latent space (sample embedding to nearest GB center embedding)
    
    Parameters:
        samples (np.ndarray): Test samples (n_samples × n_features)
        model (nn.Module): Trained CenterOnlyAutoencoder model
        gb_instances (list): List of GranularBall objects
        device (torch.device): Device (CPU/GPU) for tensor operations
        
    Returns:
        tuple: (normalized_score_r, normalized_score_l)
            - normalized_score_r: Robust-normalized reconstruction error
            - normalized_score_l: Robust-normalized latent consistency error
    """
    # Set model to evaluation mode
    model.eval()
    
    # Extract GB centers and convert to tensor
    gb_centers = np.array([gb.center for gb in gb_instances])
    if len(gb_centers) == 0:
        return np.zeros(len(samples)), np.zeros(len(samples))
    
    # Convert samples to tensor
    sample_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
    gb_center_tensor = torch.tensor(gb_centers, dtype=torch.float32).to(device)
    
    # Step 1: Find nearest GB center for each sample (original space)
    nn_finder = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(gb_centers)
    original_distances, nearest_gb_indices = nn_finder.kneighbors(samples)
    original_distances = original_distances.flatten()
    nearest_gb_indices = nearest_gb_indices.flatten()
    
    # Step 2: Compute reconstruction error (Score_R)
    with torch.no_grad():
        reconstructed_samples, sample_latents = model(sample_tensor)
        # L2 reconstruction error (mean over features)
        score_r = torch.mean((reconstructed_samples - sample_tensor) ** 2, dim=1).cpu().numpy()
        
        # Step 3: Compute latent space consistency error (Score_L)
        # Get latent embeddings of all GB centers
        _, gb_center_latents = model(gb_center_tensor)
        # Get latent embeddings of nearest GB centers
        nearest_gb_latents = gb_center_latents[nearest_gb_indices]
        # Latent space distances
        latent_distances = torch.sqrt(torch.sum((sample_latents - nearest_gb_latents) ** 2, dim=1)).cpu().numpy()
        # Consistency error (absolute difference between original and latent distances)
        score_l = np.abs(original_distances - latent_distances)
    
    # Step 4: Robust normalization of scores
    normalized_score_r = robust_normalization(score_r)
    normalized_score_l = robust_normalization(score_l)
    
    return normalized_score_r, normalized_score_l