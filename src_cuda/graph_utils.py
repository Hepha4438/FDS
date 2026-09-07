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
        F = torch.tensor(features_norm, dtype=torch.float32, device=device)
        dist_matrix = 1.0 - torch.mm(F, F.t())
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

    dist_matrix.fill_diagonal_(0.0)
    dist_matrix = dist_matrix / (dist_matrix.max() + 1e-8)

    return dist_matrix

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
    return T_constrained