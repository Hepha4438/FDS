import concurrent.futures
import threading
import logging
import os
from typing import Optional, Tuple, Dict, List
import contextlib
import anndata as ad
import numpy as np
import ot
import ot.backend as otb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import scanpy as sc
import rapids_singlecell as rsc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from model_utils
from model_utils import FeatureTransform, _to_dense_float32

# Import from graph_utils
from graph_utils import compute_knn_graph_distance, apply_cell_type_constraints

# Import from gs_utils
from gs_utils import geosketch_subsample, geosketch_target_batches, geosketch_stratified_pairing

# Import from e_step_utils and m_step_utils
from e_step_utils import compute_transport_batch
from m_step_utils import (
    aggregate_training_data_from_batches,
    train_global_model,
    apply_transfer_method,
    unstandardize_features
)

# Import from plot_utils
from plot_utils import (
    plot_dual_umap,
    plot_weight_heatmap,
    plot_convergence,
    plot_transfer_debug_umap,
    plot_spatial_channels,
    plot_spatial_mapping
)


def align_features_fgw(
    adata_a: ad.AnnData, #Source dataset
    adata_b: ad.AnnData, #Target dataset
    # Core parameters
    e_step_method: str = 'fgw',  
    m_step_method: str = 'global',  

    # Sampling strategy selection
    sampling_strategy: str = 'celltype',  

    # Celltype sampling parameters (used when sampling_strategy='celltype')
    sketch_size: int = 1000,  
    sketch_obsm_key: str = 'X_umap',
    use_stratified_pairing: bool = True,  
    stratified_pairing_fix: bool = True,  
    cell_type_col: str = 'annotation_level_0',  

    # Spatial sampling parameters (used when sampling_strategy='spatial')
    spatial_key: str = 'X_spatial',  
    window_height: Optional[float] = None,  
    window_width: Optional[float] = None,  
    window_overlap: float = 0.1,  
    spatial_knn: int = 50,  
    n_windows_target: int = 20,  

    # OT/GW parameters
    epsilon: float = 0.1,
    sinkhorn_iters: int = 1000,
    balanced_ot: bool = True,  

    use_linear_assignment: bool = False,  

    # kNN graph parameters
    knn_k: int = 30,  
    metric: str = 'cosine',  
    m_step_metric: str = 'euclidean',  
    use_knn_graph: bool = True,  

    # Fused GW parameters (for 'fgw' e_step_method)
    celltype_probs_layer: str = 'X_celltype_probs',  
    alpha: float = 0.3,  
    gamma: float = 0.4,  
    sketch_pca_components: int = 50,
    
    # Training parameters
    n_iters: int = 30,
    steps_per_iter: int = 50,  
    lr: float = 1e-3,
    weight_decay: float = 0,
    lambda_cross: float = 0.9,  
    lambda_struct: float = 0.0,  
    lambda_var: float = 0.1,  
    structure_sample_size: Optional[int] = 2048,  
    hidden_dim: Optional[int] = None,  
    dropout: float = 0.0,  
    init_strategy: str = 'auto',  
    device: Optional[str] = None,
    debug_plots_path: Optional[str] = None,
    entropy_percentile: float = 90.0,
    confidence_percentile: float = 10.0,
    verbose: bool = True,
) -> Tuple[nn.Module, np.ndarray, ad.AnnData]:
    
    # --- Device & data ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps"
    device_t = torch.device(device)
    logging.info(f"Using device: {device}")

    if debug_plots_path:
        os.makedirs(debug_plots_path, exist_ok=True)
        subdirs = ["umap", "heatmap", "umap_transfer", "convergence"]
        if sampling_strategy == 'spatial':
            subdirs.append("channels")
        for subdir in subdirs:
            os.makedirs(os.path.join(debug_plots_path, subdir), exist_ok=True)

        log_file = os.path.join(debug_plots_path, 'alignment.log')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(file_handler)

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
        logging.info(f"Logging to file: {log_file}")

    # Load data
    X_a = _to_dense_float32(adata_a.X)
    X_b = _to_dense_float32(adata_b.X)

    features_a = torch.from_numpy(X_a).to(device_t)
    features_b = torch.from_numpy(X_b).to(device_t)

    n_a_orig, d_a = features_a.shape
    n_b_orig, d_b = features_b.shape
    logging.info(f"Source (A): {n_a_orig} spots, {d_a} features. Target (B): {n_b_orig} spots, {d_b} features.")

    if sampling_strategy not in ['celltype', 'spatial']:
        raise ValueError(f"sampling_strategy must be 'celltype' or 'spatial', got: {sampling_strategy}")

    logging.info(f"=== Using {sampling_strategy.upper()} sampling strategy ===")

    # =====================================================================
    # 1. ALWAYS PREPARE STRATIFIED BATCHES FIRST 
    # (To accurately extract the 'sketch_to_original' subset for all modes)
    # =====================================================================
    if sampling_strategy == 'spatial':
        from spatial_utils import prepare_spatial_batches
        batches_strat, auxiliary_data, window_info = prepare_spatial_batches(
            adata_a, adata_b,
            spatial_key=spatial_key,
            window_height=window_height,
            window_width=window_width,
            window_overlap=window_overlap,
            n_windows_target=n_windows_target,
            spatial_knn=spatial_knn
        )
        logging.info(f"Spatial windowing: {len(batches_strat)} windows created")
        aux_features_source_strat = auxiliary_data.get('coords_a')
        aux_features_target_strat = auxiliary_data.get('coords_b')
        knn_indices_spatial = auxiliary_data.get('knn_indices')
        n_a = n_a_orig
        sketch_to_original = None
    else:  # sampling_strategy == 'celltype'
        from celltype_utils import prepare_celltype_batches
        batches_strat, auxiliary_data, sketch_to_original = prepare_celltype_batches(
            adata_a, adata_b,
            sketch_size=sketch_size,
            use_stratified_pairing=use_stratified_pairing,
            stratified_pairing_fix=stratified_pairing_fix,
            celltype_probs_layer=celltype_probs_layer,
            cell_type_col=cell_type_col,
            sketch_obsm_key=sketch_obsm_key,
            sketch_pca_components=sketch_pca_components,
            e_step_method='fgw', 
            seed=2025
        )
        n_a = auxiliary_data['n_source']
        aux_features_source_strat = auxiliary_data.get('celltype_probs_a')
        aux_features_target_strat = auxiliary_data.get('celltype_probs_b')
        knn_indices_spatial = None
        
        # Subset features_a consistently for BOTH phases if sketching is applied
        if sketch_to_original is not None:
            features_a = features_a[sketch_to_original]
            logging.info(f"Source features subsetted to {features_a.shape} for ALL phases.")

    # =====================================================================
    # 2. PREPARE FULL BATCH FOR NFGW PHASE
    # =====================================================================
    batches_full = [(np.arange(n_a), np.arange(n_b_orig))]
    aux_features_source_full = aux_features_source_strat
    aux_features_target_full = aux_features_target_strat

    # Initialize shared model and optimizer
    model = FeatureTransform(input_dim=d_a, output_dim=d_b, hidden_dim=hidden_dim, dropout=dropout).to(device_t)

    if init_strategy == 'auto':
        if sampling_strategy == 'celltype':
            model.init_identity()
            logging.info("Auto-selected identity initialization for celltype sampling")
        else:
            model.init_random()
            logging.info("Auto-selected random initialization for spatial sampling")
    elif init_strategy == 'identity':
        model.init_identity()
    elif init_strategy == 'random':
        model.init_random()
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    features_a_transformed = None
    global_feature_mean = None
    global_feature_std = None
    convergence_data = [] 
    prev_iter_mappings = {} 

    def apply_model_with_scaling(features_in: torch.Tensor) -> torch.Tensor:
        if global_feature_mean is not None and global_feature_std is not None:
            from m_step_utils import standardize_features
            features_std = standardize_features(features_in, global_feature_mean, global_feature_std)
            output_std = model(features_std)
            output = unstandardize_features(output_std, global_feature_mean, global_feature_std)
            return output
        else:
            return model(features_in)

    # =====================================================================
    # 🌟 E-M ITERATIONS (OUTER LOOP) 🌟
    # =====================================================================
    for it in tqdm(range(n_iters), desc="E-M Alignment"):
        logging.info(f"E-M Iteration {it + 1}/{n_iters}")

        # --- CURRICULUM LEARNING: PHASE SWITCH ---
        switch_iter = 5  # Cố định 5 vòng đầu tiên chạy NFGW
        
        if it < switch_iter:
            active_e_step = "nfgw"
            active_stratified = False
            active_gamma = 0.3
            current_batches = batches_full
            current_aux_src = aux_features_source_full
            current_aux_tgt = aux_features_target_full
            logging.info(f"\n--- VÒNG {it+1}: GIAI ĐOẠN 1 - MACRO NFGW (gamma={active_gamma}) ---")
        else:
            active_e_step = "fgw"
            active_stratified = True if sampling_strategy == 'celltype' and use_stratified_pairing else False
            active_gamma = 0.0
            current_batches = batches_strat
            current_aux_src = aux_features_source_strat
            current_aux_tgt = aux_features_target_strat
            logging.info(f"\n--- VÒNG {it+1}: GIAI ĐOẠN 2 - MICRO BATCHING FGW (gamma={active_gamma}) ---")

            # Resample groups logic for FGW phase
            if active_stratified and not stratified_pairing_fix:
                logging.info(f"Resampling stratified groups for iteration {it + 1} (fix=False)")
                from celltype_utils import prepare_celltype_batches
                current_batches, aux_data_new, _ = prepare_celltype_batches(
                    adata_a, adata_b,
                    sketch_size=sketch_size,
                    use_stratified_pairing=True,
                    stratified_pairing_fix=False,
                    celltype_probs_layer=celltype_probs_layer,
                    cell_type_col=cell_type_col,
                    sketch_obsm_key=sketch_obsm_key,
                    sketch_pca_components=sketch_pca_components,
                    e_step_method=active_e_step,
                    seed=2025 + it 
                )
                current_aux_src = aux_data_new.get('celltype_probs_a')
                current_aux_tgt = aux_data_new.get('celltype_probs_b')
                
                # Cập nhật cache
                batches_strat = current_batches
                aux_features_source_strat = current_aux_src
                aux_features_target_strat = current_aux_tgt
                logging.info(f"Resampled {len(current_batches)} paired groups for iteration {it + 1}")

        if m_step_method == 'transfer' and it == 0:
            features_a_transformed = features_a.clone()
            logging.info(f"Transfer method: Initialized features_a_transformed with source features (shape: {features_a.shape})")

        # ===== PHASE 1: E-STEP =====
        logging.info(f"=== E-STEP: Computing transport for all {len(current_batches)} batches ===")
        batch_results = [None] * len(current_batches)

        def process_cluster(b_idx, src_idx, tgt_idx):
            if device_t.type == 'cuda':
                stream = torch.cuda.Stream(device=device_t)
                ctx = torch.cuda.stream(stream)
            else:
                ctx = contextlib.nullcontext()
                
            with ctx:
                if it == 0:
                    features_source_base = features_a
                else:
                    if m_step_method == 'transfer':
                        features_source_base = features_a_transformed
                    else:  
                        with torch.no_grad():
                            features_source_base = apply_model_with_scaling(features_a)

                if metric == 'cosine':
                    features_s_norm = F.normalize(features_source_base, p=2, dim=1)
                    features_t_norm = F.normalize(features_b, p=2, dim=1)
                else:  
                    if global_feature_mean is not None and global_feature_std is not None:
                        from m_step_utils import standardize_features
                        features_s_norm = standardize_features(features_source_base, global_feature_mean, global_feature_std)
                        features_t_norm = standardize_features(features_b, global_feature_mean, global_feature_std)
                    else:
                        features_s_norm = features_source_base
                        features_t_norm = features_b

                knn_constraint = None
                if sampling_strategy == 'spatial' and knn_indices_spatial is not None:
                    knn_constraint = knn_indices_spatial[src_idx]

                res = compute_transport_batch(
                    source_indices=src_idx if src_idx is not None else np.arange(n_a),
                    target_indices=tgt_idx,
                    features_source_all=features_s_norm,
                    features_target_all=features_t_norm,
                    auxiliary_features_source=current_aux_src,
                    auxiliary_features_target=current_aux_tgt,  
                    gamma=active_gamma, epsilon=epsilon, metric=metric,
                    balanced=balanced_ot, use_linear_assignment=use_linear_assignment,
                    device=device_t, iteration=it, e_step_method=active_e_step,
                    entropy_percentile=entropy_percentile, confidence_percentile=confidence_percentile,
                    knn_k=knn_k, use_knn_graph=use_knn_graph, alpha=alpha, verbose=verbose,
                    knn_constraint_indices=knn_constraint, knn_penalty_weight=5.0,
                    sampling_strategy=sampling_strategy
                )
            return b_idx, res

        if device_t.type == 'cuda':
            max_workers = 1 if active_e_step == 'nfgw' else min(len(current_batches), 3)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for batch_idx, (source_batch_indices, target_batch_indices) in enumerate(current_batches):
                    futures.append(
                        executor.submit(process_cluster, batch_idx, source_batch_indices, target_batch_indices)
                    )
                
                for future in concurrent.futures.as_completed(futures):
                    b_idx, result = future.result()
                    batch_results[b_idx] = result
                    logging.info(f"  Batch {b_idx+1}/{len(current_batches)}: Transport computed (CUDA Parallel)")
        else:
            for batch_idx, (source_batch_indices, target_batch_indices) in enumerate(current_batches):
                b_idx, result = process_cluster(batch_idx, source_batch_indices, target_batch_indices)
                batch_results[b_idx] = result
                logging.info(f"  Batch {b_idx+1}/{len(current_batches)}: Transport computed (MPS Sequential)")

        for batch_idx, result in enumerate(batch_results):
            T = result['T']
            mapping_curr = T.argmax(dim=1).cpu().numpy()

            if it > 0 and batch_idx in prev_iter_mappings:
                mapping_prev = prev_iter_mappings[batch_idx]
                
                min_len = min(len(mapping_curr), len(mapping_prev))
                if min_len > 0:
                    n_changed = np.sum(mapping_curr[:min_len] != mapping_prev[:min_len])
                    pct_changed = 100.0 * n_changed / min_len
                    convergence_data.append({
                        'iter': it,
                        'batch_idx': batch_idx,
                        'n_changed': n_changed,
                        'n_total_valid': min_len,
                        'pct_changed': pct_changed
                    })
                    logging.info(f"  Convergence: Batch {batch_idx}: {n_changed}/{min_len} cells changed ({pct_changed:.1f}%)")

            prev_iter_mappings[batch_idx] = mapping_curr

        # ===== PHASE 2: M-STEP =====
        logging.info(f"=== M-STEP: Training on aggregated data from all batches ===")

        if m_step_method == 'global':
            source_agg, target_agg, agg_stats = aggregate_training_data_from_batches(
                batch_results=batch_results,
                features_source_all=features_a,
                features_target_all=features_b,
                entropy_percentile=entropy_percentile,
                confidence_percentile=confidence_percentile
            )

            step_losses, feature_mean, feature_std = train_global_model(
                model=model,
                optimizer=optimizer,
                source_features=source_agg,
                target_features=target_agg,
                steps_per_iter=steps_per_iter,
                lambda_cross=lambda_cross,
                lambda_struct=lambda_struct,
                lambda_var=lambda_var,
                metric=m_step_metric,
                structure_sample_size=structure_sample_size,
                device=device_t,
                features_target_all=features_b  
            )

            if it == 0:
                global_feature_mean = feature_mean
                global_feature_std = feature_std
            elif feature_mean is not None:
                global_feature_mean = feature_mean
                global_feature_std = feature_std

            avg_loss = np.mean(step_losses) if step_losses else 0.0
            logging.info(f"M-step: Trained on {len(source_agg)} aggregated cells, avg loss={avg_loss:.6f}")

        elif m_step_method == 'transfer':
            features_a_transformed = apply_transfer_method(
                batch_results=batch_results,
                features_target_all=features_b,
                n_source_total=n_a,
                device=device_t
            )
            logging.info(f"M-step: Applied transfer to {n_a} source cells")

        # ===== DEBUG PLOTS =====
        if debug_plots_path:
            logging.info(f"Generating debug plots for iteration {it+1}...")

            if m_step_method == 'transfer':
                a_hat_cpu = features_a_transformed.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    model.eval()  
                    a_hat_global = apply_model_with_scaling(features_a)
                    a_hat_cpu = a_hat_global.detach().cpu().numpy()

            if sketch_to_original is not None:
                obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
                obsm_dict = {}
                if spatial_key and spatial_key in adata_a.obsm:
                    obsm_dict[spatial_key] = adata_a.obsm[spatial_key][sketch_to_original]
            else:
                obs_sketched = adata_a.obs.copy()
                obsm_dict = {}
                if spatial_key and spatial_key in adata_a.obsm:
                    obsm_dict[spatial_key] = adata_a.obsm[spatial_key]
                    
            adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched, obsm=obsm_dict if obsm_dict else None)
            adata_a_transformed.obs["type"] = "source_transformed"  

            adata_b_copy = adata_b.copy()
            adata_b_copy.obs["type"] = "target"

            concat_adata_iter = ad.concat(
                [adata_a_transformed, adata_b_copy], axis=0,
                label="batch", keys=["source", "target"], index_unique="_",
            )
            
            if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
                common_ct_values = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]])
                concat_adata_iter.obs[cell_type_col] = common_ct_values.values
                
            try:
                rsc.tl.pca(concat_adata_iter)
                rsc.pp.neighbors(concat_adata_iter, use_rep='X', metric=metric)
                rsc.tl.umap(concat_adata_iter)
                logging.info(f"UMAP computation successful for iteration {it+1}")
            except Exception as e:
                logging.error(f"UMAP computation failed for iteration {it+1}: {e}")
                pass

            plot_dual_umap(
                concat_adata_iter,
                cell_type_col=cell_type_col,
                title_prefix=f'UMAP Iter {it+1}',
                save_path=os.path.join(debug_plots_path, 'umap', f"umap_iter_{it+1:04d}.png"),
                type_palette={'source_transformed': '#1f77b4', 'target': '#ff7f0e'}
            )
            concat_adata_iter.write_h5ad(f'{str(debug_plots_path)}/umap/iter{it}.h5ad')

            if sampling_strategy == 'spatial' and spatial_key is not None:
                plot_spatial_channels(
                    adata_source=adata_a_transformed,
                    adata_target=adata_b_copy,
                    spatial_key=spatial_key,
                    iteration=it+1,
                    save_path=debug_plots_path,
                    channel_idx=2  
                )
                try:
                    T_full_for_mapping = torch.zeros(n_a, n_b_orig, device=device_t, dtype=torch.float32)
                    if active_e_step == 'nfgw' or len(batch_results) == 1:
                        T_full_for_mapping = batch_results[0]['T']
                    else:
                        for result in batch_results:
                            T = result['T']
                            source_indices = result['source_indices']
                            target_indices = result['target_indices']

                            if sampling_strategy == 'spatial':
                                source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                                target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                                T_full_for_mapping[source_idx_tensor[:, None], target_idx_tensor] = T
                            elif active_stratified:
                                source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                                target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                                T_full_for_mapping[source_idx_tensor[:, None], target_idx_tensor] = T
                            else:
                                target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                                T_full_for_mapping[:, target_idx_tensor] = T

                    mapping_current = T_full_for_mapping.argmax(dim=1).cpu().numpy()
                    T_max_vals = T_full_for_mapping.max(dim=1)[0]
                    invalid_mask = (T_max_vals < 1e-6).cpu().numpy()
                    mapping_current[invalid_mask] = -1

                    coords_source = adata_a.obsm[spatial_key].astype(np.float32)
                    coords_target = adata_b.obsm[spatial_key].astype(np.float32)
                    if sketch_to_original is not None:
                        coords_source = coords_source[sketch_to_original]

                    plot_spatial_mapping(
                        coords_source=coords_source,
                        coords_target=coords_target,
                        mapping=mapping_current,
                        iteration=it+1,
                        save_path=debug_plots_path,
                        n_samples=5000
                    )
                except Exception as e:
                    logging.warning(f"Failed to create spatial mapping plot: {e}")

            if active_e_step == 'nfgw' or len(batch_results) == 1:
                T_full_all_debug = batch_results[0]['T']
            else:
                T_full_all_debug = torch.zeros(n_a, n_b_orig, device=device_t, dtype=torch.float32)
                for result in batch_results:
                    T = result['T']
                    source_indices = result['source_indices']
                    target_indices = result['target_indices']

                    if active_stratified:
                        source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                        target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                        T_full_all_debug[source_idx_tensor[:, None], target_idx_tensor] = T
                    else:
                        target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                        T_full_all_debug[:, target_idx_tensor] = T

            if use_linear_assignment and it > 0:
                T_transpose = T_full_all_debug.t()  
                target_to_source_curr = T_transpose.argmax(dim=1).cpu().numpy()  
                target_has_match_curr = (T_transpose.max(dim=1)[0] > 0.5).cpu().numpy()  

                if 'target_to_source_prev' in locals():
                    target_changed = (target_to_source_curr != target_to_source_prev) & target_has_match_curr & target_has_match_prev
                    n_target_changed = target_changed.sum()
                    n_target_matched = target_has_match_curr.sum()
                    pct_target_changed = 100.0 * n_target_changed / n_target_matched if n_target_matched > 0 else 0.0

                    convergence_data.append({
                        'iteration': it,
                        'batch_idx': 'ALL_TARGETS',  
                        'n_changed': n_target_changed,
                        'n_total_valid': n_target_matched,
                        'pct_changed': pct_target_changed
                    })
                target_to_source_prev = target_to_source_curr
                target_has_match_prev = target_has_match_curr

            if sketch_to_original is not None:
                obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
            else:
                obs_sketched = adata_a.obs.copy()

            focused_mask_full = torch.zeros(n_a, device=device_t, dtype=torch.bool)
            for result in batch_results:
                focused_mask = result.get('focused_mask', None)
                if focused_mask is not None:
                    source_indices = result['source_indices']
                    if active_stratified:
                        source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                        focused_mask_full[source_idx_tensor] = focused_mask
                    else:
                        focused_mask_full[:] = focused_mask

            plot_transfer_debug_umap(
                T_full=T_full_all_debug,
                features_b=features_b,
                adata_a_obs=obs_sketched,
                adata_b=adata_b,
                cell_type_col=cell_type_col,
                iteration=it+1,
                save_dir=os.path.join(debug_plots_path, 'umap_transfer'),
                metric='euclidean',
                focused_mask_full=focused_mask_full,  
                use_linear_assignment=use_linear_assignment  
            )

            plot_weight_heatmap(
                model=model,
                title=f'Transformation Weights Iter {it+1}',
                save_path=os.path.join(debug_plots_path, 'heatmap', f"W_iter_{it+1:04d}.png")
            )

        if 'T_full_all_debug' in locals():
            del T_full_all_debug
        if 'T_full_for_mapping' in locals():
            del T_full_for_mapping
        if 'a_hat_global' in locals():
            del a_hat_global
        if 'T_transpose' in locals():
            del T_transpose
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    logging.info("Extracting mappings from final transport plans...")
    batch_mappings = []
    for batch_idx, result in enumerate(batch_results):
        T = result['T']
        source_indices = result['source_indices']
        target_indices = result['target_indices']
        mapping_batch = T.argmax(dim=1).cpu().numpy()  
        batch_mappings.append((batch_idx, mapping_batch, target_indices))
        logging.info(f"Batch {batch_idx}: extracted mapping for {len(mapping_batch)} source cells")

    if debug_plots_path:
        plot_convergence(
            convergence_data=convergence_data,
            save_dir=os.path.join(debug_plots_path, 'convergence')
        )

    if m_step_method == 'transfer':
        logging.info("Transfer method: Using features from final iteration...")
        
        a_hat_cpu = features_a_transformed.detach().cpu().numpy()
        if sketch_to_original is not None:
            obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
        else:
            obs_sketched = adata_a.obs.copy()
        adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched)
        adata_a_transformed.obs["type"] = "source_transformed"  
        
        model = FeatureTransform(input_dim=d_a, output_dim=d_b, hidden_dim=hidden_dim, dropout=dropout).to(device_t)
        
        mapping_final = np.full(n_a, -1, dtype=np.int64)  
        
        for batch_idx, mapping_batch, target_batch_indices in batch_mappings:
            valid_mapping = mapping_batch >= 0
            if valid_mapping.any():
                valid_source_indices = np.where(valid_mapping)[0]  
                target_batch_indices_for_valid = mapping_batch[valid_mapping]  
                mapped_targets = target_batch_indices[target_batch_indices_for_valid]  
                mask = mapping_final[valid_source_indices] == -1
                mapping_final[valid_source_indices[mask]] = mapped_targets[mask]
        
        adata_b_copy = adata_b.copy()
        adata_b_copy.obs["type"] = "target"
        
        concat_adata = ad.concat(
            [adata_a_transformed, adata_b_copy], axis=0,
            label="batch", keys=["source", "target"], index_unique="_",
        )
        
        if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
            common_ct_values = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]])
            concat_adata.obs[cell_type_col] = common_ct_values.values
        
        try:
            rsc.tl.pca(concat_adata)
            rsc.pp.neighbors(concat_adata, use_rep='X', metric=metric)
            rsc.tl.umap(concat_adata)
            logging.info("Final UMAP computation successful")
        except Exception as e:
            logging.error(f"Final UMAP computation failed: {e}")

    else:  # m_step_method == 'global'
        logging.info("Combining mappings from all batches...")
        mapping_final = np.full(n_a, -1, dtype=np.int64)  

        for batch_idx, mapping_batch, target_batch_indices in batch_mappings:
            valid_mapping = mapping_batch >= 0
            if valid_mapping.any():
                valid_source_indices = np.where(valid_mapping)[0]  
                target_batch_indices_for_valid = mapping_batch[valid_mapping]  
                mapped_targets = target_batch_indices[target_batch_indices_for_valid]  
                mask = mapping_final[valid_source_indices] == -1
                mapping_final[valid_source_indices[mask]] = mapped_targets[mask]

        logging.info("Creating final concatenated data...")
        with torch.no_grad():
            model.eval()  
            a_hat_global = apply_model_with_scaling(features_a)
            a_hat_cpu = a_hat_global.detach().cpu().numpy()

        if sketch_to_original is not None:
            obs_sketched = adata_a.obs.iloc[sketch_to_original].copy()
        else:
            obs_sketched = adata_a.obs.copy()
        adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched)
        adata_a_transformed.obs["type"] = "source_transformed"  
        
        adata_b_copy = adata_b.copy()
        adata_b_copy.obs["type"] = "target"
        
        concat_adata = ad.concat(
            [adata_a_transformed, adata_b_copy], axis=0,
            label="batch", keys=["source", "target"], index_unique="_",
        )
        if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
            common_ct_values = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]])
            concat_adata.obs[cell_type_col] = common_ct_values.values
        
        try:
            rsc.tl.pca(concat_adata)
            rsc.pp.neighbors(concat_adata, use_rep='X', metric=metric)
            rsc.tl.umap(concat_adata)
        except Exception as e:
            pass
    
    if debug_plots_path:
        plot_dual_umap(
            concat_adata,
            cell_type_col=cell_type_col,
            title_prefix='Final UMAP',
            save_path=os.path.join(debug_plots_path, 'umap', 'umap_final.png'),
            type_palette={'source_transformed': '#1f77b4', 'target': '#ff7f0e'}
        )
        plot_weight_heatmap(
            model=model,
            title='Final Transformation Weights',
            save_path=os.path.join(debug_plots_path, 'heatmap', 'W_final.png')
        )
    
    logging.info(f"Alignment completed. Final mapping: {np.sum(mapping_final >= 0)}/{len(mapping_final)} source cells mapped")

    # Reconstruct full transport matrix for output
    if active_e_step == 'nfgw' or len(batch_results) == 1:
        T_full = batch_results[0]['T']
    else:
        T_full = torch.zeros(n_a, n_b_orig, device=device_t, dtype=torch.float32)
        for result in batch_results:
            T = result['T']
            source_indices = result['source_indices']
            target_indices = result['target_indices']

            if active_stratified:
                source_idx_tensor = torch.from_numpy(source_indices).long().to(device_t)
                target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                T_full[source_idx_tensor[:, None], target_idx_tensor] = T
            else:
                target_idx_tensor = torch.from_numpy(target_indices).long().to(device_t)
                T_full[:, target_idx_tensor] = T

    if True:
        from model_utils import save_alignment_model
        dir_path = "../datasets/scGPT_example/"
        model_save_path = os.path.join(dir_path, 'alignment_model.pt')
        os.makedirs(dir_path, exist_ok=True)
        save_alignment_model(
            model=model,
            save_path=model_save_path,
            feature_mean=global_feature_mean,
            feature_std=global_feature_std,
            gene_names=adata_a.var_names.tolist()
        )

    return model, mapping_final, concat_adata, T_full, global_feature_mean, global_feature_std