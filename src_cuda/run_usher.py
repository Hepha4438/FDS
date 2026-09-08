import concurrent.futures
import contextlib
import logging
import os
from typing import Optional, Tuple, Dict, List
import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

from model_utils import _to_dense_float32
from graph_utils import compute_knn_graph_distance, apply_cell_type_constraints
from e_step_utils import compute_transport_batch
from m_step_utils import train_global_model, apply_transfer_method, unstandardize_features, standardize_features
from plot_utils import plot_dual_umap, plot_weight_heatmap, plot_convergence, plot_transfer_debug_umap, plot_spatial_channels, plot_spatial_mapping

# =====================================================================
# 🌟 KIẾN TRÚC GỐC USHER: LOW-COMPLEXITY TRANSFORM 🌟
# =====================================================================
class FeatureTransform(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_residual = False
        
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, output_dim)
            nn.init.eye_(self.net.weight)
            nn.init.zeros_(self.net.bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
            # Khởi tạo gần với ma trận đơn vị để bảo tồn cấu trúc hình học ban đầu
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def align_features_fgw(
    adata_a: ad.AnnData,
    adata_b: ad.AnnData,
    e_step_method: str = 'fgw',  
    m_step_method: str = 'global',  
    sampling_strategy: str = 'celltype',  

    sketch_size: int = 1000,  
    sketch_obsm_key: str = 'X_umap',
    use_stratified_pairing: bool = True,  
    stratified_pairing_fix: bool = False,  
    cell_type_col: str = 'annotation_level_0',  

    spatial_key: str = 'X_spatial',  
    window_height: Optional[float] = None,  
    window_width: Optional[float] = None,  
    window_overlap: float = 0.1,  
    spatial_knn: int = 50,  
    n_windows_target: int = 20,  

    epsilon: float = 0.1,
    sinkhorn_iters: int = 1000,
    balanced_ot: bool = True,  
    use_linear_assignment: bool = True,  # BẮT BUỘC: Thuật toán Hungarian để tạo khung xương 1-1

    knn_k: int = 30,  
    metric: str = 'cosine',  
    m_step_metric: str = 'euclidean',  
    use_knn_graph: bool = True,  

    celltype_probs_layer: str = 'X_celltype_probs',  
    alpha: float = 0.3,  
    gamma: float = 0.4,  
    sketch_pca_components: int = 50,
    
    n_iters: int = 30,
    steps_per_iter: int = 100,  
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lambda_cross: float = 0.9,  
    lambda_struct: float = 0.0,  
    lambda_var: float = 0.1,  
    structure_sample_size: Optional[int] = 2048,  
    hidden_dim: Optional[int] = 512,  
    dropout: float = 0.0,  
    init_strategy: str = 'auto',  
    device: Optional[str] = None,
    debug_plots_path: Optional[str] = None,
    entropy_percentile: float = 90.0,
    confidence_percentile: float = 10.0,
    verbose: bool = True,
) -> Tuple[nn.Module, np.ndarray, ad.AnnData]:
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    logging.info(f"Using device: {device}")

    if debug_plots_path:
        os.makedirs(debug_plots_path, exist_ok=True)
        for subdir in ["umap", "heatmap", "umap_transfer", "convergence"] + (["channels"] if sampling_strategy == 'spatial' else []):
            os.makedirs(os.path.join(debug_plots_path, subdir), exist_ok=True)

    X_a = _to_dense_float32(adata_a.X)
    X_b = _to_dense_float32(adata_b.X)

    features_a = torch.from_numpy(X_a).to(device_t)
    features_b = torch.from_numpy(X_b).to(device_t)

    n_a_orig, d_a = features_a.shape
    n_b_orig, d_b = features_b.shape

    # =====================================================================
    # DATA BATCHING
    # =====================================================================
    if sampling_strategy == 'spatial':
        from spatial_utils import prepare_spatial_batches
        batches, auxiliary_data, window_info = prepare_spatial_batches(
            adata_a, adata_b, spatial_key=spatial_key, window_height=window_height,
            window_width=window_width, window_overlap=window_overlap,
            n_windows_target=n_windows_target, spatial_knn=spatial_knn
        )
        auxiliary_features_source = auxiliary_data.get('coords_a')
        auxiliary_features_target = auxiliary_data.get('coords_b')
        knn_indices_spatial = auxiliary_data.get('knn_indices')
        n_a = n_a_orig
        sketch_to_original = None
    else:
        from celltype_utils import prepare_celltype_batches
        batches, auxiliary_data, sketch_to_original = prepare_celltype_batches(
            adata_a, adata_b, sketch_size=sketch_size, use_stratified_pairing=use_stratified_pairing,
            stratified_pairing_fix=stratified_pairing_fix, celltype_probs_layer=celltype_probs_layer,
            cell_type_col=cell_type_col, sketch_obsm_key=sketch_obsm_key,
            sketch_pca_components=sketch_pca_components, e_step_method=e_step_method, seed=2025
        )
        n_a = auxiliary_data['n_source']
        auxiliary_features_source = auxiliary_data.get('celltype_probs_a')
        auxiliary_features_target = auxiliary_data.get('celltype_probs_b')
        knn_indices_spatial = None
        
        if sketch_to_original is not None:
            features_a = features_a[sketch_to_original]

    # Khởi tạo mô hình theo chuẩn bài báo
    model = FeatureTransform(input_dim=d_a, output_dim=d_b, hidden_dim=hidden_dim, dropout=dropout).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    features_a_transformed = None
    global_feature_mean = None
    global_feature_std = None
    convergence_data = [] 
    prev_iter_mappings = {} 

    def apply_model_with_scaling(features_in: torch.Tensor) -> torch.Tensor:
        if global_feature_mean is not None and global_feature_std is not None:
            features_std = standardize_features(features_in, global_feature_mean, global_feature_std)
            output_std = model(features_std)
            return unstandardize_features(output_std, global_feature_mean, global_feature_std)
        return model(features_in)

    # =====================================================================
    # E-M ITERATIONS
    # =====================================================================
    for it in tqdm(range(n_iters), desc="E-M Alignment (USHER Standard)"):
        
        # Resample liên tục (Stochastic Hungarian)
        if sampling_strategy == 'celltype' and use_stratified_pairing and not stratified_pairing_fix:
            from celltype_utils import prepare_celltype_batches
            batches, auxiliary_data, _ = prepare_celltype_batches(
                adata_a, adata_b, sketch_size=sketch_size, use_stratified_pairing=True,
                stratified_pairing_fix=False, celltype_probs_layer=celltype_probs_layer,
                cell_type_col=cell_type_col, sketch_obsm_key=sketch_obsm_key,
                sketch_pca_components=sketch_pca_components, e_step_method=e_step_method, seed=2025 + it 
            )
            auxiliary_features_source = auxiliary_data.get('celltype_probs_a')
            auxiliary_features_target = auxiliary_data.get('celltype_probs_b')

        if m_step_method == 'transfer' and it == 0:
            features_a_transformed = features_a.clone()

        # ===== E-STEP =====
        batch_results = [None] * len(batches)
        def process_cluster(b_idx, src_idx, tgt_idx):
            ctx = torch.cuda.stream(torch.cuda.Stream(device=device_t)) if device_t.type == 'cuda' else contextlib.nullcontext()
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
                        features_s_norm = standardize_features(features_source_base, global_feature_mean, global_feature_std)
                        features_t_norm = standardize_features(features_b, global_feature_mean, global_feature_std)
                    else:
                        features_s_norm = features_source_base
                        features_t_norm = features_b

                knn_constraint = knn_indices_spatial[src_idx] if sampling_strategy == 'spatial' and knn_indices_spatial is not None else None

                res = compute_transport_batch(
                    source_indices=src_idx if src_idx is not None else np.arange(n_a),
                    target_indices=tgt_idx,
                    features_source_all=features_s_norm, features_target_all=features_t_norm,
                    auxiliary_features_source=auxiliary_features_source,
                    auxiliary_features_target=auxiliary_features_target,  
                    gamma=gamma, epsilon=epsilon, metric=metric, balanced=balanced_ot, 
                    use_linear_assignment=use_linear_assignment, device=device_t, iteration=it, 
                    e_step_method=e_step_method, entropy_percentile=entropy_percentile, 
                    confidence_percentile=confidence_percentile, knn_k=knn_k, 
                    use_knn_graph=use_knn_graph, alpha=alpha, verbose=verbose,
                    knn_constraint_indices=knn_constraint, knn_penalty_weight=5.0,
                    sampling_strategy=sampling_strategy
                )
            return b_idx, res

        if device_t.type == 'cuda':
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
                futures = [executor.submit(process_cluster, i, src, tgt) for i, (src, tgt) in enumerate(batches)]
                for future in concurrent.futures.as_completed(futures):
                    b_idx, result = future.result()
                    batch_results[b_idx] = result
        else:
            for i, (src, tgt) in enumerate(batches):
                b_idx, result = process_cluster(i, src, tgt)
                batch_results[b_idx] = result

        for batch_idx, result in enumerate(batch_results):
            T = result['T']
            mapping_curr = T.argmax(dim=1).cpu().numpy()
            if it > 0 and batch_idx in prev_iter_mappings:
                mapping_prev = prev_iter_mappings[batch_idx]
                min_len = min(len(mapping_curr), len(mapping_prev))
                if min_len > 0:
                    n_changed = np.sum(mapping_curr[:min_len] != mapping_prev[:min_len])
                    convergence_data.append({
                        'iter': it, 'batch_idx': batch_idx, 'n_changed': n_changed,
                        'n_total_valid': min_len, 'pct_changed': 100.0 * n_changed / min_len
                    })
            prev_iter_mappings[batch_idx] = mapping_curr

        # =====================================================================
        # 🌟 M-STEP (USHER CHUẨN KẾT HỢP CHỐNG OVERCORRECTION) 🌟
        # =====================================================================
        if m_step_method == 'global':
            from m_step_utils import aggregate_training_data_from_batches
            source_agg, target_agg, agg_stats = aggregate_training_data_from_batches(
                batch_results=batch_results, features_source_all=features_a,
                features_target_all=features_b, entropy_percentile=entropy_percentile,
                confidence_percentile=confidence_percentile
            )
            
            step_losses, feature_mean, feature_std = train_global_model(
                model=model, optimizer=optimizer, source_features=source_agg,
                target_features=target_agg, steps_per_iter=steps_per_iter,
                lambda_cross=lambda_cross, lambda_struct=lambda_struct,
                lambda_var=lambda_var, metric=m_step_metric,
                structure_sample_size=structure_sample_size, device=device_t,
                features_target_all=features_b  
            )
            if it == 0 or feature_mean is not None:
                global_feature_mean = feature_mean
                global_feature_std = feature_std

        elif m_step_method == 'transfer':
            features_a_transformed = apply_transfer_method(
                batch_results=batch_results, features_target_all=features_b,
                n_source_total=n_a, device=device_t
            )

        # ===== DEBUG PLOTS =====
        if debug_plots_path:
            if m_step_method == 'transfer':
                a_hat_cpu = features_a_transformed.detach().cpu().numpy()
            else:
                with torch.no_grad():
                    model.eval()  
                    a_hat_cpu = apply_model_with_scaling(features_a).detach().cpu().numpy()

            obs_sketched = adata_a.obs.iloc[sketch_to_original].copy() if sketch_to_original is not None else adata_a.obs.copy()
            obsm_dict = {spatial_key: adata_a.obsm[spatial_key][sketch_to_original]} if spatial_key and spatial_key in adata_a.obsm and sketch_to_original is not None else {}
                    
            adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched, obsm=obsm_dict if obsm_dict else None)
            adata_a_transformed.obs["type"] = "source_transformed"  

            adata_b_copy = adata_b.copy()
            adata_b_copy.obs["type"] = "target"

            concat_adata_iter = ad.concat([adata_a_transformed, adata_b_copy], axis=0, label="batch", keys=["source", "target"], index_unique="_")
            
            if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
                concat_adata_iter.obs[cell_type_col] = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]]).values
                
            try:
                rsc.tl.pca(concat_adata_iter)
                rsc.pp.neighbors(concat_adata_iter, use_rep='X', metric=metric)
                rsc.tl.umap(concat_adata_iter)
            except Exception: pass

            plot_dual_umap(
                concat_adata_iter, cell_type_col=cell_type_col, title_prefix=f'UMAP Iter {it+1}',
                save_path=os.path.join(debug_plots_path, 'umap', f"umap_iter_{it+1:04d}.png"),
                type_palette={'source_transformed': '#1f77b4', 'target': '#ff7f0e'}
            )

        import gc; gc.collect(); torch.cuda.empty_cache()
        
    batch_mappings = [(i, r['T'].argmax(dim=1).cpu().numpy(), r['target_indices']) for i, r in enumerate(batch_results)]

    if debug_plots_path:
        plot_convergence(convergence_data=convergence_data, save_dir=os.path.join(debug_plots_path, 'convergence'))

    # Final transformation
    with torch.no_grad():
        model.eval()
        a_hat_cpu = apply_model_with_scaling(features_a).detach().cpu().numpy()

    obs_sketched = adata_a.obs.iloc[sketch_to_original].copy() if sketch_to_original is not None else adata_a.obs.copy()
    adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched)
    adata_a_transformed.obs["type"] = "source_transformed"  
    
    adata_b_copy = adata_b.copy()
    adata_b_copy.obs["type"] = "target"
    concat_adata = ad.concat([adata_a_transformed, adata_b_copy], axis=0, label="batch", keys=["source", "target"], index_unique="_")
    if cell_type_col in adata_a_transformed.obs.columns and cell_type_col in adata_b_copy.obs.columns:
        concat_adata.obs[cell_type_col] = pd.concat([adata_a_transformed.obs[cell_type_col], adata_b_copy.obs[cell_type_col]]).values
    
    try:
        rsc.tl.pca(concat_adata); rsc.pp.neighbors(concat_adata, use_rep='X', metric=metric); rsc.tl.umap(concat_adata)
    except Exception: pass

    T_full = torch.zeros(n_a, n_b_orig, device=device_t, dtype=torch.float32)
    for result in batch_results:
        if use_stratified_pairing:
            T_full[torch.from_numpy(result['source_indices']).long().to(device_t)[:, None], torch.from_numpy(result['target_indices']).long().to(device_t)] = result['T']
        else:
            T_full[:, torch.from_numpy(result['target_indices']).long().to(device_t)] = result['T']

    if True:
        from model_utils import save_alignment_model
        dir_path = "../datasets/scGPT_example/"
        os.makedirs(dir_path, exist_ok=True)
        save_alignment_model(
            model=model, 
            save_path=os.path.join(dir_path, 'alignment_model.pt'), 
            feature_mean=global_feature_mean, 
            feature_std=global_feature_std, 
            gene_names=adata_a.var_names.tolist()
        )

    return model, np.full(n_a, -1, dtype=np.int64), concat_adata, T_full, global_feature_mean, global_feature_std