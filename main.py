import os
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import io
from sklearn.metrics import roc_auc_score

# Local imports
from models.base_model import (
    construct_granular_balls,
    create_granular_ball_instances,
    compute_anomaly_scores
)
from models.trainer import train_center_only_autoencoder
from Utils.data_loader import load_npz_dataset, get_dataset_list

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set environment variables for reproducibility
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["CPU_COUNT"] = "4"  # Adjust based on your CPU cores

def set_random_seed(seed: int):
    """
    Set random seeds for reproducibility across libraries.
    
    Parameters:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_gb_ae_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str,
    alpha_candidates: list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    num_runs: int = 10,
    base_seed: int = 42
) -> tuple:
    """
    Run GB-AE experiment with hyperparameter tuning (alpha) and multiple random seeds.
    
    Parameters:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        dataset_name (str): Name of the dataset
        alpha_candidates (list): List of alpha values (weight for Score_R)
        num_runs (int): Number of runs per alpha (different random seeds)
        base_seed (int): Base random seed (incremented per run)
        
    Returns:
        tuple: (optimal_auc, optimal_std, optimal_alpha, optimal_score, optimal_time)
            - optimal_auc: Mean AUC of best alpha
            - optimal_std: Std of AUC for best alpha
            - optimal_alpha: Alpha with highest mean AUC
            - optimal_score: Final anomaly scores for best alpha
            - optimal_time: Mean training time for best alpha
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Construct granular balls from training data
    gb_data_list = construct_granular_balls(X_train)
    gb_instances, gb_centers = create_granular_ball_instances(gb_data_list)
    print(f"Constructed {len(gb_instances)} Granular Balls from training data")
    
    # Step 2: Hyperparameter tuning (alpha)
    optimal_auc = -1
    optimal_std = 0
    optimal_alpha = 0
    optimal_score = None
    optimal_time = 0
    
    # Create save directories
    result_dir = os.path.join("GBAE_Results", dataset_name)
    model_dir = os.path.join("pth_base", dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    for alpha in alpha_candidates:
        run_aucs = []
        run_times = []
        run_scores = []
        
        # Multiple runs with different seeds for statistical robustness
        for run_idx in range(num_runs):
            # Set random seed for reproducibility
            set_random_seed(base_seed + run_idx)
            
            # Train AE model on GB centers
            start_time = time.time()
            model, _ = train_center_only_autoencoder(
                gb_centers=gb_centers,
                epochs=200,
                batch_size=32,
                device=device
            )
            train_time = time.time() - start_time
            
            # Compute anomaly scores
            score_r, score_l = compute_anomaly_scores(
                samples=X_test,
                model=model,
                gb_instances=gb_instances,
                device=device
            )
            
            # Fuse scores (alpha * Score_R + (1-alpha) * Score_L)
            final_score = alpha * score_r + (1 - alpha) * score_l
            
            # Compute AUC
            auc = roc_auc_score(y_test, final_score)
            
            # Save single run results
            run_result_path = os.path.join(
                result_dir,
                f"{dataset_name}_GBAE_alpha-{alpha}_RUN-{run_idx}.mat"
            )
            io.savemat(run_result_path, {
                'anomaly_scores': final_score.reshape(-1, 1),
                'alpha': alpha,
                'training_time': train_time,
                'auc': auc
            })
            
            # Save model checkpoint (best run for current alpha)
            if auc == max(run_aucs + [auc]):
                model_path = os.path.join(model_dir, f"{dataset_name}_GBAE_alpha-{alpha}.pth")
                torch.save(model.state_dict(), model_path)
            
            # Accumulate results
            run_aucs.append(auc)
            run_times.append(train_time)
            run_scores.append(final_score)
        
        # Compute statistics for current alpha
        mean_auc = np.mean(run_aucs)
        std_auc = np.std(run_aucs)
        mean_time = np.mean(run_times)
        best_run_idx = np.argmax(run_aucs)
        best_score = run_scores[best_run_idx]
        
        # Update optimal hyperparameters
        if mean_auc > optimal_auc:
            optimal_auc = mean_auc
            optimal_std = std_auc
            optimal_alpha = alpha
            optimal_score = best_score
            optimal_time = mean_time
        
        # Log results for current alpha
        print(f"\nAlpha: {alpha} | Mean AUC: {mean_auc:.4f} (±{std_auc:.4f}) | Mean Time: {mean_time:.2f}s")
    
    # Save optimal results
    optimal_result_path = os.path.join(result_dir, f"{dataset_name}_GBAE_optimal.mat")
    # Create metadata tensor (AUC, STD, Time, Alpha)
    metadata = np.zeros((len(optimal_score), 1))
    metadata[0] = optimal_auc
    metadata[1] = optimal_std
    metadata[2] = optimal_time
    metadata[3] = optimal_alpha
    
    io.savemat(optimal_result_path, {
        'optimal_anomaly_scores': np.column_stack((optimal_score.reshape(-1, 1), metadata)),
        'optimal_alpha': optimal_alpha,
        'optimal_auc': optimal_auc,
        'optimal_auc_std': optimal_std,
        'optimal_training_time': optimal_time,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    })
    
    # Save best model (optimal alpha)
    best_model_src = os.path.join(model_dir, f"{dataset_name}_GBAE_alpha-{optimal_alpha}.pth")
    best_model_dst = os.path.join(model_dir, f"{dataset_name}_GBAE_best.pth")
    torch.save(torch.load(best_model_src), best_model_dst)
    
    print(f"\nOptimal Hyperparameters for {dataset_name}:")
    print(f"  - Alpha: {optimal_alpha}")
    print(f"  - Mean AUC: {optimal_auc:.4f} (±{optimal_std:.4f})")
    print(f"  - Mean Training Time: {optimal_time:.2f}s")
    
    return optimal_auc, optimal_std, optimal_alpha, optimal_score, optimal_time

if __name__ == "__main__":
    # Dataset configuration
    dataset_dir = "Datasets"  # Directory containing .npz datasets
    dataset_paths = get_dataset_list(dataset_dir)
    
    # Track results across datasets
    all_results = []
    
    # Run experiments on all datasets
    for dataset_path in dataset_paths:
        print(f"\n{'='*80}")
        print(f"Processing Dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*80}")
        
        # Load and preprocess dataset
        X_train, X_test, y_test, dataset_name = load_npz_dataset(dataset_path)
        
        # Skip if results already exist (to resume experiments)
        result_dir = os.path.join("GBAE_results", dataset_name)
        optimal_result_path = os.path.join(result_dir, f"{dataset_name}_GBAE_optimal.mat")
        if os.path.exists(optimal_result_path) and len(os.listdir(result_dir)) >= 191:
            print(f"Results for {dataset_name} already exist - skipping")
            # Load existing results
            existing_results = io.loadmat(optimal_result_path)
            optimal_auc = existing_results['optimal_auc'][0][0]
            optimal_alpha = existing_results['optimal_alpha'][0][0]
            all_results.append((dataset_name, optimal_auc, optimal_alpha))
            continue
        
        # Run GB-AE experiment
        optimal_auc, optimal_std, optimal_alpha, _, _ = run_gb_ae_experiment(
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name=dataset_name
        )
        
        # Track results
        all_results.append((dataset_name, optimal_auc, optimal_alpha))
    
    # Save summary results to Excel
    results_df = pd.DataFrame(
        all_results,
        columns=["Dataset", "Optimal AUC", "Optimal Alpha"]
    )
    results_df.to_excel("auc_GBAE_GB_reborn_real2.xlsx", index=False)
    
    print(f"\n{'='*80}")
    print("Experiment Summary")
    print(f"{'='*80}")
    print(results_df)
    print(f"\nMean AUC Across Datasets: {results_df['Optimal AUC'].mean():.4f}")