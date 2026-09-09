"""
M-step utilities for learning feature transformations.

This module provides functions to train models or apply transfer methods
using transport plans from the E-step.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_feature_scaler(
    features_target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_mean = features_target.mean(dim=0, keepdim=True)
    feature_std = features_target.std(dim=0, keepdim=True)
    feature_std = torch.clamp(feature_std, min=1e-6)
    logging.info(f"  [Feature Scaler] Computed from target:")
    logging.info(f"    Mean per feature: min={feature_mean.min():.6f}, max={feature_mean.max():.6f}, avg={feature_mean.mean():.6f}")
    logging.info(f"    Std per feature: min={feature_std.min():.6f}, max={feature_std.max():.6f}, avg={feature_std.mean():.6f}")
    return feature_mean, feature_std

def standardize_features(
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor
) -> torch.Tensor:
    return (features - feature_mean) / feature_std

def unstandardize_features(
    features_standardized: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor
) -> torch.Tensor:
    return features_standardized * feature_std + feature_mean

def select_focused_cells(
    T: torch.Tensor,
    entropy_percentile: float = 30.0,
    confidence_percentile: float = 70.0
) -> Tuple[torch.Tensor, Dict]:
    n_source, n_target = T.shape
    T_safe = T.clamp(min=1e-10)
    entropy = -(T * torch.log(T_safe)).sum(dim=1)
    max_probs = T.max(dim=1)[0]
    entropy_threshold = torch.quantile(entropy, entropy_percentile / 100.0)
    confidence_threshold = torch.quantile(max_probs, confidence_percentile / 100.0)
    low_entropy_mask = entropy <= entropy_threshold
    high_confidence_mask = max_probs >= confidence_threshold
    focused_mask = low_entropy_mask & high_confidence_mask
    n_focused = focused_mask.sum().item()
    n_total = len(focused_mask)
    stats = {
        'n_focused': n_focused,
        'n_total': n_total,
        'pct_focused': 100 * n_focused / n_total if n_total > 0 else 0.0,
        'entropy_threshold': entropy_threshold.item(),
        'confidence_threshold': confidence_threshold.item(),
        'avg_entropy': entropy[focused_mask].mean().item() if n_focused > 0 else 0.0,
        'avg_confidence': max_probs[focused_mask].mean().item() if n_focused > 0 else 0.0
    }
    return focused_mask, stats

def aggregate_training_data_from_batches(
    batch_results: List[Dict],
    features_source_all: torch.Tensor,
    features_target_all: torch.Tensor,
    entropy_percentile: float = 30.0,
    confidence_percentile: float = 70.0
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    all_source_features = []
    all_target_features = []
    total_focused = 0
    total_cells = 0
    batch_stats = []

    for batch_idx, result in enumerate(batch_results):
        T = result['T']
        source_indices = result['source_indices']
        target_indices = result['target_indices']

        focused_mask = result.get('focused_mask', None)
        if focused_mask is not None:
            n_focused = focused_mask.sum().item()
            n_total = len(focused_mask)
            stats = {'n_focused': n_focused, 'n_total': n_total, 'pct_focused': 100 * n_focused / n_total if n_total > 0 else 0.0, 'precomputed': True}
        else:
            focused_mask, stats = select_focused_cells(T, entropy_percentile, confidence_percentile)
            stats['precomputed'] = False
            n_focused = focused_mask.sum().item()

        if n_focused == 0:
            batch_stats.append(stats)
            continue

        focused_local_indices = torch.where(focused_mask)[0]
        focused_global_indices = source_indices[focused_local_indices.cpu().numpy()]

        device = features_source_all.device
        focused_global_idx_tensor = torch.from_numpy(focused_global_indices).long().to(device)
        source_features_batch = features_source_all[focused_global_idx_tensor]

        best_targets_local = T.argmax(dim=1)[focused_mask]
        best_targets_global = target_indices[best_targets_local.cpu().numpy()]

        best_targets_global_tensor = torch.from_numpy(best_targets_global).long().to(device)
        target_features_batch = features_target_all[best_targets_global_tensor]

        all_source_features.append(source_features_batch)
        all_target_features.append(target_features_batch)

        total_focused += n_focused
        total_cells += T.shape[0]
        batch_stats.append(stats)

    if len(all_source_features) == 0:
        raise ValueError("No focused cells found in any batch! Cannot train model.")

    source_features_aggregated = torch.cat(all_source_features, dim=0)
    target_features_aggregated = torch.cat(all_target_features, dim=0)

    aggregation_stats = {
        'total_focused': total_focused,
        'total_cells': total_cells,
        'pct_focused': 100 * total_focused / total_cells if total_cells > 0 else 0.0,
        'n_batches_used': len(all_source_features),
        'n_batches_total': len(batch_results),
        'batch_stats': batch_stats
    }
    return source_features_aggregated, target_features_aggregated, aggregation_stats

def train_global_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    steps_per_iter: int,
    lambda_cross: float,
    lambda_var: float,
    metric: str,
    device: torch.device,
    features_target_all: Optional[torch.Tensor] = None
) -> List[float]:
    """
    Train global transformation model (USHER Original Standard - No Structural Loss).
    """
    model.train()
    torch.set_grad_enabled(True)

    step_losses = []

    # ===== FEATURE STANDARDIZATION =====
    feature_mean = None
    feature_std = None
    if metric == 'euclidean':
        if features_target_all is not None:
            feature_mean, feature_std = compute_feature_scaler(features_target_all)
        else:
            feature_mean, feature_std = compute_feature_scaler(target_features)

        source_features = standardize_features(source_features, feature_mean, feature_std)
        target_features = standardize_features(target_features, feature_mean, feature_std)

    logging.info("  [M-step] Chạy Adam Optimizer (Bản gốc USHER)...")
    for step in range(steps_per_iter):
        optimizer.zero_grad()

        # Forward pass
        source_transformed = model(source_features)
        
        # ===== Loss 1: Cross-domain alignment =====
        if metric == 'cosine':
            source_norm = F.normalize(source_transformed, p=2, dim=1)
            target_norm = F.normalize(target_features, p=2, dim=1)
            loss_cross = (1.0 - (source_norm * target_norm).sum(dim=1)).mean()
        else:  
            loss_cross = F.mse_loss(source_transformed, target_features)
        
        # ===== Loss 2: Variance preservation =====
        loss_var = (source_transformed.std(dim=0) - target_features.std(dim=0)).abs().mean()

        # ===== Combined loss (Chỉ còn 2 thành phần) =====
        loss = lambda_cross * loss_cross + lambda_var * loss_var

        if not torch.isfinite(loss):
            logging.error(f"Non-finite loss at step {step}: loss={loss.item()}")
            break

        loss.backward()
        optimizer.step()

        step_losses.append(loss.item())

        if (step + 1) % 10 == 0 or step == 0:
            logging.info(f"  Step {step+1}/{steps_per_iter}: loss={loss.item():.6f} (cross={loss_cross.item():.4f}, var={loss_var.item():.4f})")

    if step_losses:
        logging.info(f"M-step completed: {len(step_losses)} steps, avg loss = {np.mean(step_losses):.6f}")

    return step_losses, feature_mean, feature_std

def apply_transfer_method(
    batch_results: List[Dict],
    features_target_all: torch.Tensor,
    n_source_total: int,
    device: torch.device
) -> torch.Tensor:
    n_target = features_target_all.shape[0]
    T_full = torch.zeros(n_source_total, n_target, device=device, dtype=torch.float32)

    for result in batch_results:
        T = result['T']
        source_indices = result['source_indices']
        target_indices = result['target_indices']

        source_idx_tensor = torch.from_numpy(source_indices).long().to(device)
        target_idx_tensor = torch.from_numpy(target_indices).long().to(device)

        T_full[source_idx_tensor[:, None], target_idx_tensor] = T

    features_source_transformed = T_full @ features_target_all

    nnz = (T_full > 1e-6).sum().item()
    nnz_pct = 100 * nnz / T_full.numel()
    logging.info(f"Transfer method: T_full nnz = {nnz}/{T_full.numel()} ({nnz_pct:.2f}%)")
    logging.info(f"Transfer method: Transformed {n_source_total} source cells")

    return features_source_transformed