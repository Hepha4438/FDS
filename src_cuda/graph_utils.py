"""
Graph utilities for spatial alignment.
"""

import logging
from typing import Optional
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


def compute_knn_graph_distance(
    features: torch.Tensor,
    k: int = 30,
    metric: str = 'cosine',
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Computes topology-preserving graph distances using Softmin Landmark Geodesic.
    """
    if device is None:
        device = features.device

    features_np = features.cpu().numpy()
    n_samples = features_np.shape[0]
    k = min(k, n_samples - 1)

    features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
    if metric == 'cosine':
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine', algorithm='brute').fit(features_norm)
        distances, indices = nbrs.kneighbors(features_norm)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='auto').fit(features_np)
        distances, indices = nbrs.kneighbors(features_np)

    row_indices = np.repeat(np.arange(n_samples), k)
    col_indices = indices[:, 1:].flatten()
    edge_weights = np.ones(len(row_indices), dtype=np.float32)

    adjacency_matrix = csr_matrix((edge_weights, (row_indices, col_indices)), shape=(n_samples, n_samples))
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

    try:
        logging.info("Computing Fast Softmin Landmark Geodesic Approximation...")
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin

        num_landmarks = min(512, n_samples)
        
        if n_samples > num_landmarks * 10:
            num_candidates = num_landmarks * 10
            candidate_indices = np.random.choice(n_samples, num_candidates, replace=False)
            candidate_features = features_np[candidate_indices]
            
            kmeans = KMeans(n_clusters=num_landmarks, n_init=1, random_state=42)
            kmeans.fit(candidate_features)
            
            closest_in_candidates = pairwise_distances_argmin(kmeans.cluster_centers_, candidate_features)
            landmarks = candidate_indices[closest_in_candidates]
        else:
            landmarks = np.random.choice(n_samples, num_landmarks, replace=False)

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
        E = torch.exp(-beta * L_dist)
        M = torch.mm(E, E.t())
        M = torch.clamp(M, min=1e-30) 
        dist_matrix = - (1.0 / beta) * torch.log(M)
        
    except Exception as e:
        logging.warning(f"Softmin computation failed: {e}. Fallback to direct metric.")
        F = torch.tensor(features_norm, dtype=torch.float32, device=device)
        dist_matrix = 1.0 - torch.mm(F, F.t())
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

    dist_matrix.fill_diagonal_(0.0)
    dist_matrix = dist_matrix / (dist_matrix.max() + 1e-8)

    return dist_matrix


def extract_topological_signatures(
    features: torch.Tensor,
    k: int = 30,
    metric: str = 'cosine',
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Trích xuất "Chữ ký hình thái" (Topological Signatures) không giám sát 
    để tạo M_topo cho quá trình Unsupervised Warm-start.
    """
    if device is None:
        device = features.device

    features_np = features.cpu().numpy()
    n_samples = features_np.shape[0]
    k_eff = min(k, n_samples - 1)

    features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
    if metric == 'cosine':
        nbrs = NearestNeighbors(n_neighbors=k_eff+1, metric='cosine', algorithm='brute').fit(features_norm)
        distances, indices = nbrs.kneighbors(features_norm)
    else:
        nbrs = NearestNeighbors(n_neighbors=k_eff+1, metric=metric, algorithm='auto').fit(features_np)
        distances, indices = nbrs.kneighbors(features_np)

    # 1. Tính In-Degree Centrality
    col_indices = indices[:, 1:].flatten()
    in_degrees = np.bincount(col_indices, minlength=n_samples).astype(np.float32)

    # 2. Xây dựng đồ thị vô hướng để duyệt BFS
    row_indices = np.repeat(np.arange(n_samples), k_eff)
    edge_weights = np.ones(len(row_indices), dtype=np.float32)
    
    adjacency_matrix = csr_matrix((edge_weights, (row_indices, col_indices)), shape=(n_samples, n_samples))
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

    num_landmarks = min(512, n_samples)
    landmarks = np.random.choice(n_samples, num_landmarks, replace=False)

    landmark_dists = shortest_path(adjacency_matrix, directed=False, indices=landmarks, unweighted=True)
    Max_dist = np.nanmax(landmark_dists[landmark_dists != np.inf])
    if np.isnan(Max_dist): Max_dist = 1.0
    landmark_dists[np.isinf(landmark_dists)] = Max_dist
    
    L_dist = torch.tensor(landmark_dists.T, dtype=torch.float32, device=device)
    beta = 2.0  
    E = torch.exp(-beta * L_dist)

    # 3. Tính các mô-men thống kê phân phối hình học
    E_mean = E.mean(dim=1)
    E_var = E.var(dim=1, unbiased=False)
    E_std = torch.sqrt(E_var + 1e-8)
    E_skew = ((E - E_mean.unsqueeze(1))**3).mean(dim=1) / (E_std**3)
    
    in_degrees_t = torch.tensor(in_degrees, dtype=torch.float32, device=device)
    
    # Kết hợp thành signature (N, 4)
    signatures = torch.stack([in_degrees_t, E_mean, E_var, E_skew], dim=1)
    
    # Chuẩn hóa Z-score nội bộ
    sig_mean = signatures.mean(dim=0, keepdim=True)
    sig_std = signatures.std(dim=0, keepdim=True) + 1e-8
    signatures = (signatures - sig_mean) / sig_std
    
    return signatures


def apply_cell_type_constraints(
    T: torch.Tensor,
    cell_types_source: np.ndarray,
    cell_types_target: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    n_source, n_target = T.shape
    cell_types_source_expanded = np.tile(cell_types_source.reshape(-1, 1), (1, n_target))
    cell_types_target_expanded = np.tile(cell_types_target.reshape(1, -1), (n_source, 1))
    same_type_mask = (cell_types_source_expanded == cell_types_target_expanded)
    same_type_mask_tensor = torch.from_numpy(same_type_mask).to(device)

    T_constrained = T * same_type_mask_tensor
    row_sums_original = T.sum(dim=1, keepdim=True)
    row_sums_constrained = T_constrained.sum(dim=1, keepdim=True)
    row_sums_constrained = row_sums_constrained.clamp(min=1e-10)

    T_constrained = T_constrained * (row_sums_original / row_sums_constrained)

    n_matches_before = (T > 1e-6).sum().item()
    n_matches_after = (T_constrained > 1e-6).sum().item()
    n_zeroed = n_matches_before - n_matches_after

    if n_matches_before > 0:
        logging.info(f"Cell type constraints: {n_zeroed}/{n_matches_before} matches zeroed out ({100*n_zeroed/n_matches_before:.1f}%)")
    else:
        logging.info(f"Cell type constraints: No matches in transport plan")

    return T_constrained