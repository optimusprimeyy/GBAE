# Granular-Ball Autoencoder-Based Anomalies (GBAE)
This repository implements a Granular Ball (GB)-based Autoencoder (AE) framework for unsupervised anomaly detection, integrating granular computing principles with deep learning to enhance detection performance on high-dimensional datasets.

## Methodology Overview
1. **Granular Ball Construction**: Partition training data into granular balls using a divisive clustering strategy (2-means) with statistical distance (SD) minimization criterion.
2. **Center-Only Autoencoder**: Train an AE exclusively on GB centers to capture global data structure while reducing computational complexity.
3. **Anomaly Scoring**: Combine reconstruction error (Score_R) and latent space consistency error (Score_L) to compute final anomaly scores.

## Experimental Setup
### Datasets
- Place datasets in `Datasets/` (NPZ format with `x_train`, `x_test`, `y_test` keys)
- Each NPZ file should contain:
  - `x_train`: Training features (n_samples × n_features)
  - `x_test`: Test features (n_samples × n_features)
  - `y_test`: Test labels (1 = anomaly, 0 = normal)

### Environment Setup
```bash
pip install -r requirements.txt