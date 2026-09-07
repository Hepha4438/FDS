import concurrent.futures
import contextlib
import logging
import os
from typing import Optional, Tuple, Dict
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
from m_step_utils import apply_transfer_method, unstandardize_features
from plot_utils import plot_dual_umap, plot_weight_heatmap, plot_convergence, plot_transfer_debug_umap, plot_spatial_channels, plot_spatial_mapping

# =====================================================================
# 🌟 KIẾN TRÚC MỚI: CONDITIONAL FLOW MATCHING (CFM) VECTOR FIELD
# =====================================================================
class FlowMatchingVectorField(nn.Module):
    """
    Mạng MLP học trường vận tốc v_theta(x_t, t) để dẫn dắt tế bào 
    từ không gian Xenium (t=0) sang scRNA-seq (t=1) theo quỹ đạo mượt mà.
    """
    def __init__(self, dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        # Nhập vào đặc trưng x_t (dim) và thời gian t (1 chiều) -> Tổng dim + 1
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        # Khởi tạo trọng số tiến sát 0 để ở bước đầu tiên mô hình đóng vai trò như Identity mapping (giữ nguyên không gian)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t có shape (batch_size, 1), nối trực tiếp vào biểu diễn x
        if t.ndim == 1:
            t = t.unsqueeze(1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def train_flow_matching_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    steps_per_iter: int,
    device: torch.device,
) -> List[float]:
    """
    Huấn luyện trường vận tốc bằng hàm mất mát Conditional Flow Matching (CFM).
    """
    model.train()
    step_losses = []
    batch_size_cfm = min(2048, source_features.shape[0])

    for step in range(steps_per_iter):
        optimizer.zero_grad()

        # Lấy ngẫu nhiên một batch các cặp giả lập từ E-step (x0: source, x1: target)
        indices = torch.randint(0, source_features.shape[0], (batch_size_cfm,), device=device)
        x0 = source_features[indices]
        x1 = target_features[indices]

        # Lấy mẫu thời gian ngẫu nhiên t từ phân phối đều U[0, 1]
        t = torch.rand(batch_size_cfm, 1, device=device)

        # Tạo điểm trên đường thẳng nối (Interpolation path)
        xt = (1.0 - t) * x0 + t * x1

        # Vận tốc mục tiêu chuẩn xác (Target velocity) của đường thẳng nối
        v_target = x1 - x0

        # Dự đoán vận tốc từ mô hình mạng MLP
        v_pred = model(xt, t)

        # Hàm mất mát Conditional Flow Matching (MSE giữa vận tốc dự đoán và thực tế)
        loss = F.mse_loss(v_pred, v_target)

        if not torch.isfinite(loss):
            break

        loss.backward()
        optimizer.step()
        step_losses.append(loss.item())

    return step_losses


def apply_flow_ode_solver(
    model: nn.Module,
    features: torch.Tensor,
    n_steps: int = 10
) -> torch.Tensor:
    """
    Giải phương trình vi phân thường (Euler ODE Solver) từ t = 0 đến t = 1 
    để dịch chuyển toàn bộ tế bào Xenium sang không gian scRNA-seq một cách mượt mà.
    """
    model.eval()
    with torch.no_grad():
        x = features.clone()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.full((x.shape[0], 1), t_val, device=features.device)
            v = model(x, t_tensor)
            x = x + v * dt
    return x


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
    use_linear_assignment: bool = True,  # Mặc định bật Hungarian cho CFM

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
    dropout: float = 0.1,  
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

    # Khởi tạo mô hình Flow Matching Vector Field thay thế FeatureTransform cũ
    model = FlowMatchingVectorField(dim=d_a, hidden_dim=hidden_dim, dropout=dropout).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    features_a_transformed = None
    global_feature_mean = None
    global_feature_std = None
    convergence_data = [] 
    prev_iter_mappings = {} 

    # =====================================================================
    # E-M ITERATIONS
    # =====================================================================
    for it in tqdm(range(n_iters), desc="E-M Alignment (Flow Matching)"):
        
        # Resample for micro-mixing
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

        # ===== E-STEP (Chuẩn USHER gốc) =====
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
                            features_source_base = apply_flow_ode_solver(model, features_a, n_steps=5)

                if metric == 'cosine':
                    features_s_norm = F.normalize(features_source_base, p=2, dim=1)
                    features_t_norm = F.normalize(features_b, p=2, dim=1)
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
        # 🌟 M-STEP: HUẤN LUYỆN TRƯỜNG VẬN TỐC CONDITIONAL FLOW MATCHING (CFM)
        # =====================================================================
        if m_step_method == 'global':
            from m_step_utils import aggregate_training_data_from_batches
            source_agg, target_agg, agg_stats = aggregate_training_data_from_batches(
                batch_results=batch_results, features_source_all=features_a,
                features_target_all=features_b, entropy_percentile=entropy_percentile,
                confidence_percentile=confidence_percentile
            )
            
            # Huấn luyện mô hình Flow Matching Vector Field
            step_losses = train_flow_matching_model(
                model=model, optimizer=optimizer,
                source_features=source_agg, target_features=target_agg,
                steps_per_iter=steps_per_iter, device=device_t
            )

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
                    a_hat_cpu = apply_flow_ode_solver(model, features_a, n_steps=10).detach().cpu().numpy()

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

    # Final transformation using ODE solver
    with torch.no_grad():
        a_hat_cpu = apply_flow_ode_solver(model, features_a, n_steps=20).detach().cpu().numpy()

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

    return model, np.full(n_a, -1, dtype=np.int64), concat_adata, T_full, None, None