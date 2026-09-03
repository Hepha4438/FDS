import time
import torch
import numpy as np
import logging
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from typing import Optional
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from sklearn.datasets import make_blobs 
from sklearn.datasets import make_swiss_roll

# =====================================================================
# OLD FUNCTION (EXACT GEODESIC / DIJKSTRA)
# =====================================================================
def compute_knn_graph_distance_old(features: torch.Tensor, k: int = 5, metric: str = 'cosine', device: Optional[str] = None) -> torch.Tensor:
    features_np = features.cpu().numpy()
    n_samples = features_np.shape[0]
    k = min(k, n_samples - 1)
    
    features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute').fit(features_norm)
    distances, indices = nbrs.kneighbors(features_np)

    row_indices, col_indices, edge_weights = [], [], []
    for i in range(n_samples):
        for j in range(1, k+1):
            neighbor_idx = indices[i, j]
            row_indices.extend([i, neighbor_idx])
            col_indices.extend([neighbor_idx, i])
            edge_weights.extend([1.0, 1.0])

    adjacency_matrix = csr_matrix((edge_weights, (row_indices, col_indices)), shape=(n_samples, n_samples))
    dist_matrix = shortest_path(adjacency_matrix, directed=False, unweighted=False, return_predecessors=False)
    
    dist_matrix[dist_matrix > 1e6] = np.nanmax(dist_matrix[dist_matrix != np.inf])
    dist_matrix = dist_matrix / dist_matrix.max()
    return torch.from_numpy(dist_matrix.astype(np.float32))

# =====================================================================
# NEW FUNCTION (LANDMARK GEODESIC APPROXIMATION)
# =====================================================================
def compute_knn_graph_distance_new(
    features: torch.Tensor,
    k: int = 30,
    metric: str = 'cosine',
    device: Optional[str] = None,
) -> torch.Tensor:
    if device is None:
        device = features.device

    features_np = features.cpu().numpy()
    n_samples = features_np.shape[0]
    k = min(k, n_samples - 1)

    # 1. VECTORIZED k-NN
    features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute').fit(features_norm)
    distances, indices = nbrs.kneighbors(features_np)

    row_indices = np.repeat(np.arange(n_samples), k)
    col_indices = indices[:, 1:].flatten()
    edge_weights = np.ones(len(row_indices), dtype=np.float32)

    adjacency_matrix = csr_matrix((edge_weights, (row_indices, col_indices)), shape=(n_samples, n_samples))
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

    try:
        # 2. LANDMARKS
        num_landmarks = min(512, n_samples)
        landmarks = np.random.choice(n_samples, num_landmarks, replace=False)

        # 3. EXACT BFS TỪ CÁC TRẠM TRUNG CHUYỂN
        landmark_dists = shortest_path(
            adjacency_matrix,
            directed=False,
            indices=landmarks,
            unweighted=True
        )

        Max_dist = np.nanmax(landmark_dists[landmark_dists != np.inf])
        if np.isnan(Max_dist): Max_dist = 1.0
        landmark_dists[np.isinf(landmark_dists)] = Max_dist
        
        L_dist = torch.tensor(landmark_dists.T, dtype=torch.float32, device=device)
        beta = 2.0
        
        # Bước 1: Exponentiation E = exp(-beta * L_dist)
        E = torch.exp(-beta * L_dist)
        
        # Bước 2: Matrix Multiplication E @ E.T
        # Đây chính là trái tim của Min-Plus nhưng được biên dịch qua C++/Metal của PyTorch
        M = torch.mm(E, E.t())
        
        # Chống lỗi Log(0)
        M = torch.clamp(M, min=1e-30)
        
        # Bước 3: Logarithm để khôi phục khoảng cách Geodesic gốc
        dist_matrix = - (1.0 / beta) * torch.log(M)
        
    except Exception as e:
        logging.warning(f"Softmin computation failed: {e}")
        F = torch.tensor(features_norm, dtype=torch.float32, device=device)
        dist_matrix = 1.0 - torch.mm(F, F.t())
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

    # Đảm bảo đường chéo bằng 0
    dist_matrix.fill_diagonal_(0.0)

    # Chuẩn hóa về [0, 1] cho USHER
    dist_matrix = dist_matrix / (dist_matrix.max() + 1e-8)

    return dist_matrix

# =====================================================================
# BENCHMARK SCENARIO 1: CONTINUOUS MANIFOLD MOCK DATA
# =====================================================================
def run_performance_test(n_samples=10000, n_features=512, k=30):
    print(f"Starting benchmark with {n_samples} cells, {n_features} dimensions, kNN={k}...")
    
    # 1. Tạo đa tạp cuộn 3D - Đặc trưng cho quỹ đạo phân nhánh liên tục
    print("Generating Continuous Swiss Roll Manifold (Biological Trajectory Proxy)...")
    X_3d, _ = make_swiss_roll(n_samples=n_samples, noise=0.5, random_state=42)
    
    # 2. Nhúng (Embed) đa tạp 3D vào không gian 512D bằng ma trận chiếu ngẫu nhiên
    np.random.seed(42)
    projection_matrix = np.random.randn(3, n_features)
    X_numpy = np.dot(X_3d, projection_matrix)
    
    # 3. L2 Normalization (Mô phỏng chính xác X_scGPT embeddings)
    X_numpy = X_numpy / np.linalg.norm(X_numpy, axis=1, keepdims=True)
    features_tensor = torch.tensor(X_numpy, dtype=torch.float32)

    # TEST 1 (GEODESIC BFS CŨ)
    start_time = time.time()
    D_old = compute_knn_graph_distance_old(features_tensor, k=k, metric='cosine')
    time_old = time.time() - start_time
    print(f"⏱️ [Old] Geodesic execution time: {time_old:.4f} seconds")

    # TEST 2 (SOFTMIN LANDMARK MỚI)
    start_time = time.time()
    D_new = compute_knn_graph_distance_new(features_tensor, k=k, metric='cosine')
    time_new = time.time() - start_time
    print(f"⏱️ [New] Softmin Landmark execution time: {time_new:.4f} seconds")
    
    print("\nComputing structural correlation (Spearman Rank)...")
    D_old_flat = D_old.cpu().numpy().flatten()
    D_new_flat = D_new.cpu().numpy().flatten()
    
    correlation, p_value = spearmanr(D_old_flat, D_new_flat)
    
    print("="*50)
    print("📊 CORE BENCHMARK RESULTS:")
    print("="*50)
    print(f"🚀 Speedup: {time_old / time_new:.2f}x faster")
    print(f"🎯 Topology Preservation (Spearman): {correlation:.4f} (Expected > 0.85)")
    print("="*50)

# =====================================================================
# BENCHMARK SCENARIO 2: REAL DATA
# =====================================================================
def run_test_on_real_data(X_real, k=30, max_samples=6000):
    print("\n==================================================")
    print("🔬 TESTING ON REAL DATA")
    print("==================================================")
    
    if X_real.shape[0] > max_samples:
        print(f"⚠️ Original data has {X_real.shape[0]} cells. Randomly subsampling {max_samples} cells to prevent Geodesic bottleneck...")
        np.random.seed(42)
        indices = np.random.choice(X_real.shape[0], max_samples, replace=False)
        X_subset = X_real[indices]
    else:
        X_subset = X_real
        
    features_tensor = torch.tensor(X_subset, dtype=torch.float32)
    device = next(model.parameters()).device if 'model' in globals() else 'cpu'
    features_tensor = features_tensor.to(device)

    print(f"\nRunning Geodesic (O(N^2)) on {max_samples} cells...")
    start_time = time.time()
    D_old = compute_knn_graph_distance_old(features_tensor, k=k, metric='cosine')
    time_old = time.time() - start_time
    print(f"⏱️ [Old] Time: {time_old:.4f} seconds")

    print(f"\nRunning Landmark Geodesic (O(N)) on {max_samples} cells...")
    start_time = time.time()
    D_new = compute_knn_graph_distance_new(features_tensor, k=k, metric='cosine')
    time_new = time.time() - start_time
    print(f"⏱️ [New] Time: {time_new:.4f} seconds")
    
    print("\nComputing structural correlation (Spearman Rank)...")
    D_old_flat = D_old.cpu().numpy().flatten()
    D_new_flat = D_new.cpu().numpy().flatten()
    
    correlation, _ = spearmanr(D_old_flat, D_new_flat)
    
    print("\n" + "="*50)
    print("📊 REAL DATA BENCHMARK RESULTS:")
    print("="*50)
    print(f"🚀 Speedup                          : {time_old / time_new:.2f}x faster")
    print(f"🎯 Topology Preservation (Spearman) : {correlation:.4f} (Expected > 0.85)")
    print("="*50)

    sample_plot = np.random.choice(len(D_old_flat), size=5000, replace=False)
    plt.figure(figsize=(8, 6))
    plt.scatter(D_old_flat[sample_plot], D_new_flat[sample_plot], alpha=0.3, s=2, color='royalblue')
    plt.title(f"Structural Correlation on Real Data\n(Spearman: {correlation:.4f})")
    plt.xlabel("Exact Geodesic Distance (Normalized)")
    plt.ylabel("Landmark Geodesic Distance (Normalized)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

import scanpy as sc
import os

# =====================================================================
# LOAD REAL DATA FROM DISK
# =====================================================================
xenium_file_path = "/Users/HIEU/Workspaces/Projects/FDS/datasets/xenium_scGPT.h5ad"

if not os.path.exists(xenium_file_path):
    raise FileNotFoundError(f"❌ Data file not found at: {xenium_file_path}. Please check the path!")

print(f"Loading Xenium data from {xenium_file_path}...")
xenAD_scGPT = sc.read_h5ad(xenium_file_path)

X_real_data = xenAD_scGPT.obsm['X_scGPT']

run_performance_test(n_samples=7500, n_features=512, k=30)
run_test_on_real_data(X_real_data, k=30, max_samples=7500)