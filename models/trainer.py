import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# Import CenterOnlyAutoencoder from base_model (resolve circular import)
from .base_model import CenterOnlyAutoencoder
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=1e-6, smooth_window=3):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.smooth_window = smooth_window  # 滑动窗口大小
        self.loss_history = []  # 保存最近的损失值

    def __call__(self, val_loss):
        # 滑动平均平滑损失
        self.loss_history.append(val_loss)
        if len(self.loss_history) > self.smooth_window:
            self.loss_history.pop(0)
        smooth_loss = np.mean(self.loss_history)
        
        score = -smooth_loss  # 最小化损失 → 最大化score
        if self.best_score is None:
            self.best_score = score
            self.counter = 0  # 重置计数器
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[EarlyStopping] counter: {self.counter}/{self.patience} (no improvement)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'[EarlyStopping] Loss improved (smooth loss: {smooth_loss:.6f}), reset counter to 0')

def train_center_only_autoencoder(
    gb_centers: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    hidden_dim: int = None,
    patience: int = 15,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.3,
    device: torch.device = None
) -> tuple:
    """
    Train CenterOnlyAutoencoder on Granular Ball centers with early stopping.
    
    Training Objective: Minimize reconstruction loss of GB centers (L2 loss).
    
    Parameters:
        gb_centers (np.ndarray): Matrix of GB centers (n_gb × n_features)
        epochs (int): Maximum training epochs
        batch_size (int): Batch size (clamped to number of GB centers if smaller)
        hidden_dim (int): Latent space dimension (default: max(2, input_dim//4))
        patience (int): Early stopping patience
        learning_rate (float): Adam optimizer learning rate
        weight_decay (float): L2 regularization strength
        dropout_rate (float): Dropout probability for AE layers
        device (torch.device): Training device (CPU/GPU, default: auto-detect)
        
    Returns:
        tuple: (trained_model, training_losses)
            - trained_model: Trained CenterOnlyAutoencoder model
            - training_losses: List of epoch-wise training losses
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get input dimension and number of GB centers
    input_dim = gb_centers.shape[1]
    num_centers = len(gb_centers)
    # Clamp batch size to number of centers (avoid empty batches)
    batch_size = min(batch_size, num_centers)
    
    # Convert GB centers to tensor
    gb_center_tensor = torch.tensor(gb_centers, dtype=torch.float32).to(device)
    
    # Initialize model, optimizer, and early stopping
    model = CenterOnlyAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    training_losses = []
    
    # Log training configuration
    print(f"\nTraining Configuration:")
    print(f"  - GB Centers: {num_centers} | Input Dimension: {input_dim}")
    print(f"  - Batch Size: {batch_size} | Device: {device}")
    print(f"  - Max Epochs: {epochs} | Early Stopping Patience: {patience}")
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        epoch_loss = 0.0
        # Random permutation for batch shuffling
        permutation = torch.randperm(num_centers)
        
        # Mini-batch training
        for i in range(0, num_centers, batch_size):
            # Get batch indices and batch data
            batch_indices = permutation[i:i+batch_size]
            batch_centers = gb_center_tensor[batch_indices]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed_centers, _ = model(batch_centers)
            
            # Compute reconstruction loss (L2)
            recon_loss = torch.mean((reconstructed_centers - batch_centers) ** 2)
            
            # Backward pass and optimization
            recon_loss.backward()
            optimizer.step()
            
            # Accumulate epoch loss
            epoch_loss += recon_loss.item() * len(batch_centers)
        
        # Normalize epoch loss by number of centers
        epoch_loss /= num_centers
        training_losses.append(epoch_loss)
        
        # Check early stopping
        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at Epoch {epoch+1}")
            break
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 and not early_stopping.early_stop:
            print(f"\nEpoch [{epoch+1}/{epochs}] | Reconstruction Loss: {epoch_loss:.6f}")
    
    return model, training_losses

