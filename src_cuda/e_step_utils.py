"""
E-step utilities for optimal transport computation.
"""

import logging
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
import ot
import pandas as pd
import ot.backend as otb
ot.backend._BACKEND_IMPLEMENTATIONS = [
    b for b in ot.backend._BACKEND_IMPLEMENTATIONS 
    if b.__name__ != 'JaxBackend'
]
from graph_utils import compute_knn_graph_distance


def compute_transport_ot(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    aux_features_source: Optional[np.ndarray],
    aux_features_target: Optional[np.ndarray],
    gamma: float,
    epsilon: float,
    metric: str,
    balanced: bool,
    device: torch.device,
    iteration: int = 0,
    knn_mask: Optional[torch.Tensor] = None,
    knn_penalty: float = 5.0,
    warmstart_duals: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    verbose: bool = False
) -> torch.Tensor:
    
    n_source = features_source.shape[0]
    n_target = features_target.shape[0]
    gamma_effective = 1.0 if iteration == 0 else gamma

    if aux_features_source is not None and aux_features_target is not None:
        aux_source_torch = torch.from_numpy(aux_features_source).to(device).float()
        aux_target_torch = torch.from_numpy(aux_features_target).to(device).float()
        M_aux = torch.cdist(aux_source_torch, aux_target_torch, p=2)

        if torch.isnan(M_aux).any():
            logging.warning("M_aux contains NaN values.")
            
        M_aux = M_aux / (M_aux.max().clamp(min=1e-8))
        if torch.isnan(features_source).any():
            logging.warning("features_source contains NaN values.")
            
        if metric == 'cosine':
            M_features = 1.0 - (features_source @ features_target.T)
        else: 
            M_features = torch.cdist(features_source, features_target, p=2)
            
        M_features = M_features / (M_features.max().clamp(min=1e-8))
        M = (gamma_effective) * M_aux + (1-gamma_effective) * M_features
    else:
        if metric == 'cosine':
            M = 1.0 - (features_source @ features_target.T)
        else:
            M = torch.cdist(features_source, features_target, p=2)
        M = M / (M.mean().clamp(min=1e-8))

    if knn_mask is not None:
        penalty_value = M[knn_mask].median() * knn_penalty 
        M = M.clone() 
        M[~knn_mask] += penalty_value

    p = torch.ones(n_source, device=device, dtype=torch.float32) / n_source
    q = torch.ones(n_target, device=device, dtype=torch.float32) / n_target

    backend = otb.TorchBackend()
    if balanced:
        if epsilon < 0.05:
            T = ot.bregman.sinkhorn_log(p, q, M, reg=epsilon, numItermax=1000, backend=backend)
        else:
            T = ot.bregman.sinkhorn(p, q, M, reg=epsilon, numItermax=1000, backend=backend)
        if not isinstance(T, torch.Tensor):
            T = torch.as_tensor(T, device=device).float()
    else:
        T = ot.unbalanced.sinkhorn_unbalanced(
            p.cpu().numpy(), q.cpu().numpy(), M.cpu().numpy(),
            reg=epsilon, reg_m=0.8, numItermax=1000
        )
        T = torch.from_numpy(T).to(device).float()

    row_sums = T.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums
    if torch.isnan(T).any():
        T = torch.nan_to_num(T, nan=0.0)

    return T


def compute_transport_gw(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    epsilon: float,
    knn_k: int,
    metric: str,
    use_knn_graph: bool,
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:
    
    n_source = features_source.shape[0]
    n_target = features_target.shape[0]

    if use_knn_graph:
        C1 = compute_knn_graph_distance(features_source, k=knn_k, metric=metric, device=device)
        C2 = compute_knn_graph_distance(features_target, k=knn_k, metric=metric, device=device)
    else:
        C1 = torch.cdist(features_source, features_source, p=2)
        C1 = C1 / (C1.max() + 1e-8)
        C2 = torch.cdist(features_target, features_target, p=2)
        C2 = C2 / (C2.max() + 1e-8)

    p = torch.ones(n_source, device=device, dtype=torch.float32) / n_source
    q = torch.ones(n_target, device=device, dtype=torch.float32) / n_target

    T = ot.gromov.entropic_gromov_wasserstein(
        C1=C1, C2=C2, p=p, q=q, loss_fun='square_loss',
        epsilon=epsilon, max_iter=10000, tol=1e-6,
        verbose=verbose, log=False, backend='torch'
    )

    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, device=device).float()

    row_sums = T.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums

    return T


def compute_transport_fgw(
    features_source: torch.Tensor,
    features_target: torch.Tensor,
    aux_features_source: Optional[np.ndarray],
    aux_features_target: Optional[np.ndarray],
    gamma: float,
    epsilon: float,
    alpha: float,
    metric: str,
    knn_k: int,
    use_knn_graph: bool,
    device: torch.device,
    iteration: int = 0,
    verbose: bool = False
) -> torch.Tensor:
    
    n_source = features_source.shape[0]
    n_target = features_target.shape[0]
    gamma_effective = 1.0 if iteration == 0 else gamma

    if use_knn_graph:
        C1 = compute_knn_graph_distance(features_source, k=knn_k, metric=metric, device=device)
        C2 = compute_knn_graph_distance(features_target, k=knn_k, metric=metric, device=device)
    else:
        C1 = torch.cdist(features_source, features_source, p=2)
        C1 = C1 / (C1.mean() + 1e-8)
        C2 = torch.cdist(features_target, features_target, p=2)
        C2 = C2 / (C2.mean() + 1e-8)

    if aux_features_source is not None and aux_features_target is not None:
        aux_source_torch = torch.from_numpy(aux_features_source).float().to(device)
        aux_target_torch = torch.from_numpy(aux_features_target).float().to(device)
        M_aux = torch.cdist(aux_source_torch, aux_target_torch, p=2)
        M_aux = M_aux / (M_aux.max().clamp(min=1e-8))
        
        if metric == 'cosine':
            M_features = 1.0 - (features_source @ features_target.T)
        else:
            M_features = torch.cdist(features_source, features_target, p=2)
        M_features = M_features / (M_features.max().clamp(min=1e-8))

        M = (gamma_effective) * M_aux + (1-gamma_effective) * M_features
    else:
        if metric == 'cosine':
            M = 1.0 - (features_source @ features_target.T)
        else:
            M = torch.cdist(features_source, features_target, p=2)
        M = M / (M.max().clamp(min=1e-8))

    p = torch.ones(n_source, device=device, dtype=torch.float32) / n_source
    q = torch.ones(n_target, device=device, dtype=torch.float32) / n_target

    T = ot.gromov.entropic_fused_gromov_wasserstein(
        M=M, C1=C1, C2=C2, p=p, q=q, loss_fun='square_loss',
        epsilon=epsilon, alpha=alpha, max_iter=10000,
        tol=1e-6, verbose=verbose, log=False, backend='torch'
    )

    if not isinstance(T, torch.Tensor):
        T = torch.as_tensor(T, device=device).float()

    row_sums = T.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums

    return T


def apply_linear_assignment(
    T: torch.Tensor,
    gamma_effective: float,
    e_step_method: str,
    M: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    from scipy.optimize import linear_sum_assignment
    cost_matrix = -T.cpu().numpy()  
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    T_sparse = torch.zeros_like(T)
    T_sparse[row_indices, col_indices] = 1.0
    return T_sparse, row_indices, col_indices


def compute_transport_batch(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    features_source_all: torch.Tensor,
    features_target_all: torch.Tensor,
    auxiliary_features_source: Optional[np.ndarray] = None,
    auxiliary_features_target: Optional[np.ndarray] = None,
    gamma: float = 0.5,
    epsilon: float = 0.1,
    metric: str = 'cosine',
    balanced: bool = True,
    use_linear_assignment: bool = False,
    device: torch.device = None,
    iteration: int = 0,
    e_step_method: str = 'ot',
    entropy_percentile: float = 60.0,
    confidence_percentile: float = 40.0,
    knn_k: int = 15,
    use_knn_graph: bool = True,
    alpha: float = 0.5,
    verbose: bool = False,
    knn_constraint_indices: Optional[np.ndarray] = None,
    knn_penalty_weight: float = 5.0,
    warmstart_duals: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    sampling_strategy: str = 'celltype'
) -> Dict:
    
    if device is None:
        device = features_source_all.device

    source_idx_tensor = torch.from_numpy(source_indices).long().to(device)
    target_idx_tensor = torch.from_numpy(target_indices).long().to(device)

    features_source = features_source_all[source_idx_tensor]
    features_target = features_target_all[target_idx_tensor]

    aux_features_source = None
    aux_features_target = None
    if auxiliary_features_source is not None:
        if isinstance(auxiliary_features_source, pd.DataFrame):
            src_arr = auxiliary_features_source.to_numpy()
        else:
            src_arr = np.array(auxiliary_features_source)
        aux_features_source = src_arr[source_indices]
        
    if auxiliary_features_target is not None:
        if isinstance(auxiliary_features_target, pd.DataFrame):
            tgt_arr = auxiliary_features_target.to_numpy()
        else:
            tgt_arr = np.array(auxiliary_features_target)
        aux_features_target = tgt_arr[target_indices]

    knn_mask = None
    if knn_constraint_indices is not None and sampling_strategy == 'spatial':
        n_source_batch = features_source.shape[0]
        n_target_batch = features_target.shape[0]
        knn_mask = torch.zeros(n_source_batch, n_target_batch, device=device, dtype=torch.bool)
        for i in range(n_source_batch):
            knn_targets = knn_constraint_indices[i]
            for target_idx in knn_targets:
                if target_idx in target_indices:
                    local_idx = np.where(target_indices == target_idx)[0]
                    if len(local_idx) > 0:
                        knn_mask[i, local_idx[0]] = True

    if e_step_method == 'ot':
        T = compute_transport_ot(
            features_source, features_target, aux_features_source, aux_features_target,
            gamma, epsilon, metric, balanced, device, iteration,
            knn_mask=knn_mask, knn_penalty=knn_penalty_weight,
            warmstart_duals=warmstart_duals, verbose=verbose
        )
    elif e_step_method == 'gw':
        T = compute_transport_gw(
            features_source, features_target, epsilon=epsilon, knn_k=knn_k,
            metric=metric, use_knn_graph=use_knn_graph, device=device, verbose=verbose
        )
    elif e_step_method == 'fgw':
        T = compute_transport_fgw(
            features_source, features_target, aux_features_source, aux_features_target,
            gamma=gamma, epsilon=epsilon, alpha=alpha, metric=metric, knn_k=knn_k,
            use_knn_graph=use_knn_graph, device=device, iteration=iteration, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown e_step_method: {e_step_method}.")

    n_source = T.shape[0]
    entropy = torch.zeros(n_source, device=device, dtype=torch.float32)
    max_probs = torch.zeros(n_source, device=device, dtype=torch.float32)
    
    chunk_size_ent = 5000
    for st in range(0, n_source, chunk_size_ent):
        en = min(st + chunk_size_ent, n_source)
        T_chunk = T[st:en]
        T_safe_chunk = T_chunk.clamp(min=1e-10)
        
        entropy[st:en] = -(T_chunk * torch.log(T_safe_chunk)).sum(dim=1)
        max_probs[st:en] = T_chunk.max(dim=1)[0]

    entropy_threshold = torch.quantile(entropy, entropy_percentile / 100.0)
    confidence_threshold = torch.quantile(max_probs, confidence_percentile / 100.0)

    low_entropy_mask = entropy <= entropy_threshold  
    high_confidence_mask = max_probs >= confidence_threshold  
    focused_mask = low_entropy_mask & high_confidence_mask

    row_indices = None
    col_indices = None
    if use_linear_assignment:
        gamma_effective = 1.0 if iteration == 0 else gamma
        T, row_indices, col_indices = apply_linear_assignment(
            T, gamma_effective, e_step_method, M=None
        )

    return {
        'T': T,
        'source_indices': source_indices,
        'target_indices': target_indices,
        'row_indices': row_indices,
        'col_indices': col_indices,
        'focused_mask': focused_mask
    }