with open('DATASET_NOTE.md', 'a') as f:
    f.write('''

## Phase 6: Graph Autoencoder (GAE)

### 1. Input Node Features
- Dimension: 3
- Features: `[x, y, speed]`
- Preprocessing: StandardScaling fitted per-window.

### 2. Graph Representation
- Edges based on 200m physical communication range, encoded as PyTorch Geometric `edge_index` (undirected, therefore symmetric entries in `edge_index`).

### 3. Encoder Architecture
- Framework: PyTorch Geometric
- Layers: 2x GCNConv
- Layer Dimensions: `3 -> 64 -> 16`
- Activation: ReLU after first layer.

### 4. Latent Dimension
- `Z` belongs to R^16.

### 5. Decoder
- Inner-product decoder computing `A_hat = sigmoid(Z Z^T)` to reconstruct graph topology.

### 6. Reconstruction Target
- Original unweighted VANET adjacency matrix.

### 7. Reconstruction Loss
- Binary Cross Entropy (BCE) over all pairs of nodes.
- A positive edge weight `pos_weight` is dynamically calculated per window to combat sparsity (`pos_weight = negative_edges / positive_edges`).

### 8. Positive-Class Handling
- Evaluated using ROC-AUC and Average Precision (ignoring self-loops) directly on the decoder output against actual edges.

### 9. Optimizer
- Adam Optimizer.

### 10. Learning Rate
- 0.01 with weight decay of 0.0001.

### 11. Epochs
- 200 epochs maximum per window.

### 12. Early Stopping
- Triggered if reconstruction BCE loss does not improve for 20 epochs.

### 13. Random Seed
- Set to 42 for NumPy and PyTorch (CPU and CUDA).

### 14. Clustering Procedure
- K-Means with K=5, `n_init=20`, `random_seed=42`, applied natively to the learned 16D latent embeddings.

### 15. Evaluation Metrics
- Modularity and internal-edge ratio against the original graph.
- Silhouette, Davies-Bouldin, and Calinski-Harabasz evaluated in the latent space.

### 16. Temporal-Overlap Preparation
- Conducted a forward scan matching cluster overlap across `t` and `t+1` windows using Jaccard Similarity to prepare for Hungarian matching.

### 17. Limitations
- The inner product decoder forces an O(N^2) dense reconstruction matrix. While computationally fine for current N~500 networks, highly dense environments will mandate negative sampling optimization.
''')
