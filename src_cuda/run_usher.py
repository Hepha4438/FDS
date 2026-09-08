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
from plot_utils import plot_dual_umap, plot_weight_heatmap, plot_convergence, plot_transfer_debug_umap

# =====================================================================
# 🌟 KIẾN TRÚC MỚI: AdaLN + SCHRÖDINGER BRIDGES (SDE)
# =====================================================================
class MixtureFlowMatchingVectorField(nn.Module):
    """
    Kiến trúc Flow Matching sử dụng Adaptive Layer Normalization (AdaLN).
    Ép Mạng Neural phải 'xé rách' không gian bằng cách dùng ngữ cảnh cụm (c) 
    và thời gian (t) để can thiệp trực tiếp vào phân phối của từng layer.
    """
    def __init__(self, dim: int, num_mixtures: int, context_dim: int = 128, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.input_dim = dim
        self.output_dim = dim
        self.hidden_dim = hidden_dim
        self.use_residual = False  # Biến này để tương thích với hàm save_alignment_model
        
        # Mạng nội suy Thời gian và Cụm riêng biệt
        self.time_mlp = nn.Sequential(
            nn.Linear(1, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim)
        )
        self.mixture_embed = nn.Embedding(num_mixtures, context_dim)
        
        # Layer chiếu dữ liệu
        self.x_proj = nn.Linear(dim, hidden_dim)
        
        # Mạng AdaLN (Tạo tham số Scale và Shift từ điều kiện)
        self.cond_proj1 = nn.Linear(context_dim * 2, hidden_dim * 2)
        self.cond_proj2 = nn.Linear(context_dim * 2, hidden_dim * 2)
        
        # LayerNorm không có tham số học (sẽ được AdaLN điều khiển)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Khởi tạo ZERO cho tính ổn định của ODE/SDE
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)
        nn.init.zeros_(self.cond_proj1.weight)
        nn.init.zeros_(self.cond_proj1.bias)
        nn.init.zeros_(self.cond_proj2.weight)
        nn.init.zeros_(self.cond_proj2.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
            
        # 1. Trích xuất nhúng ngữ cảnh tổng hợp
        t_emb = self.time_mlp(t)
        c_emb = self.mixture_embed(c)
        cond = torch.cat([t_emb, c_emb], dim=1)  # Shape: (batch, context_dim * 2)
        
        # 2. Sinh ra Scale và Shift cho từng block
        scale1, shift1 = self.cond_proj1(cond).chunk(2, dim=1)
        scale2, shift2 = self.cond_proj2(cond).chunk(2, dim=1)
        
        # 3. Block 1 với AdaLN
        h = self.x_proj(x)
        h_norm = self.norm1(h)
        h = h_norm * (1.0 + scale1) + shift1  # Tác động lực xé cụm tại đây
        h = self.act(h)
        h = self.lin1(h)
        h = self.dropout(h)
        
        # 4. Block 2 với AdaLN
        h_norm = self.norm2(h)
        h = h_norm * (1.0 + scale2) + shift2  # Tác động lực xé cụm lần 2
        h = self.act(h)
        out = self.lin2(h)
        
        return out


def train_schrodinger_bridge_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_contexts: torch.Tensor,
    steps_per_iter: int,
    device: torch.device,
    sigma: float = 0.2
) -> List[float]:
    model.train()
    step_losses = []
    batch_size = min(2048, source_features.shape[0])

    for step in range(steps_per_iter):
        optimizer.zero_grad()

        # Lấy mẫu mỏ neo
        indices = torch.randint(0, source_features.shape[0], (batch_size,), device=device)
        x0 = source_features[indices]
        x1 = target_features[indices]
        c = source_contexts[indices]

        # Lấy mẫu t ngẫu nhiên (Uniform distribution)
        t = torch.rand(batch_size, 1, device=device)
        
        # 🌟 MA THUẬT SDE: Bơm nhiễu Brownian (Brownian Bridge)
        noise = torch.randn_like(x0)
        # Nhiễu đạt cực đại ở giữa chặng (t=0.5) và triệt tiêu ở 2 đầu (t=0 và t=1)
        std = sigma * torch.sqrt(t * (1.0 - t)) 
        xt = (1.0 - t) * x0 + t * x1 + std * noise

        v_target = x1 - x0
        v_pred = model(xt, t, c)

        loss = F.mse_loss(v_pred, v_target)

        if not torch.isfinite(loss):
            break

        loss.backward()
        optimizer.step()
        step_losses.append(loss.item())

    return step_losses


def apply_sb_sde_solver(
    model: nn.Module,
    features: torch.Tensor,
    contexts: torch.Tensor,
    n_steps: int = 40,
    sigma: float = 0.2
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = features.clone()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = step * dt
            t_tensor = torch.full((x.shape[0], 1), t_val, device=features.device)
            
            # Tính lực đẩy (Drift) từ mạng AdaLN
            v = model(x, t_tensor, contexts)
            
            # 🌟 MA THUẬT SDE: Bơm nhiễu Euler-Maruyama Method
            dw = torch.randn_like(x) * np.sqrt(dt)
            # Tắt nhiễu ở 5 bước cuối để tế bào hạ cánh chính xác vào tâm
            current_sigma = sigma if step < (n_steps - 5) else 0.0
            
            x = x + v * dt + current_sigma * dw
            
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
    use_linear_assignment: bool = True,

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
        for subdir in ["umap", "heatmap", "umap_transfer", "convergence"]:
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

    # Bản đồ ngữ cảnh
    source_to_context = torch.zeros(n_a, dtype=torch.long, device=device_t)
    for b_idx, (src, _) in enumerate(batches):
        source_to_context[src] = b_idx

    model = MixtureFlowMatchingVectorField(dim=d_a, num_mixtures=len(batches), hidden_dim=hidden_dim, dropout=dropout).to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    features_a_transformed = None
    convergence_data = [] 
    prev_iter_mappings = {} 

    # =====================================================================
    # E-M ITERATIONS
    # =====================================================================
    for it in tqdm(range(n_iters), desc="E-M Alignment (Schrödinger Bridge)"):
        
        if use_stratified_pairing and not stratified_pairing_fix:
            batches, auxiliary_data, _ = prepare_celltype_batches(
                adata_a, adata_b, sketch_size=sketch_size, use_stratified_pairing=True,
                stratified_pairing_fix=False, celltype_probs_layer=celltype_probs_layer,
                cell_type_col=cell_type_col, sketch_obsm_key=sketch_obsm_key,
                sketch_pca_components=sketch_pca_components, e_step_method=e_step_method, seed=2025 + it 
            )
            auxiliary_features_source = auxiliary_data.get('celltype_probs_a')
            auxiliary_features_target = auxiliary_data.get('celltype_probs_b')
            for b_idx, (src, _) in enumerate(batches):
                source_to_context[src] = b_idx

        # ===== E-STEP =====
        batch_results = [None] * len(batches)
        def process_cluster(b_idx, src_idx, tgt_idx):
            ctx = torch.cuda.stream(torch.cuda.Stream(device=device_t)) if device_t.type == 'cuda' else contextlib.nullcontext()
            with ctx:
                if it == 0:
                    features_source_base = features_a
                else:
                    with torch.no_grad():
                        # Dùng bộ giải SDE để dịch chuyển tế bào
                        features_source_base = apply_sb_sde_solver(model, features_a, source_to_context, n_steps=5, sigma=0.2)

                if metric == 'cosine':
                    features_s_norm = F.normalize(features_source_base, p=2, dim=1)
                    features_t_norm = F.normalize(features_b, p=2, dim=1)
                else:  
                    features_s_norm = features_source_base
                    features_t_norm = features_b

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
                    knn_constraint_indices=None, knn_penalty_weight=5.0,
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
        # 🌟 M-STEP: CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN SB-SDE
        # =====================================================================
        source_list = []
        target_list = []
        context_list = []

        from m_step_utils import select_focused_cells
        for batch_idx, result in enumerate(batch_results):
            T = result['T']
            src_idx = result['source_indices']
            tgt_idx = result['target_indices']

            focused_mask = result.get('focused_mask', None)
            if focused_mask is None:
                focused_mask, _ = select_focused_cells(T, entropy_percentile, confidence_percentile)

            if focused_mask.sum().item() == 0: continue

            best_targets_local = T.argmax(dim=1)[focused_mask]
            
            focused_global_idx = src_idx[torch.where(focused_mask)[0].cpu().numpy()]
            best_targets_global = tgt_idx[best_targets_local.cpu().numpy()]

            source_list.append(features_a[torch.from_numpy(focused_global_idx).long().to(device_t)])
            target_list.append(features_b[torch.from_numpy(best_targets_global).long().to(device_t)])
            
            context_list.append(torch.full((len(focused_global_idx),), batch_idx, dtype=torch.long, device=device_t))

        if len(source_list) > 0:
            source_agg = torch.cat(source_list, dim=0)
            target_agg = torch.cat(target_list, dim=0)
            context_agg = torch.cat(context_list, dim=0)

            # Huấn luyện Schrödinger Bridge SDE với nhiễu
            step_losses = train_schrodinger_bridge_model(
                model=model, optimizer=optimizer, source_features=source_agg,
                target_features=target_agg, source_contexts=context_agg,
                steps_per_iter=steps_per_iter, device=device_t, sigma=0.2
            )

        # ===== DEBUG PLOTS =====
        if debug_plots_path:
            with torch.no_grad():
                model.eval()  
                a_hat_cpu = apply_sb_sde_solver(model, features_a, source_to_context, n_steps=10, sigma=0.2).detach().cpu().numpy()

            obs_sketched = adata_a.obs.iloc[sketch_to_original].copy() if sketch_to_original is not None else adata_a.obs.copy()
            adata_a_transformed = ad.AnnData(X=a_hat_cpu, obs=obs_sketched)
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
                plot_dual_umap(
                    concat_adata_iter, cell_type_col=cell_type_col, title_prefix=f'UMAP Iter {it+1}',
                    save_path=os.path.join(debug_plots_path, 'umap', f"umap_iter_{it+1:04d}.png"),
                    type_palette={'source_transformed': '#1f77b4', 'target': '#ff7f0e'}
                )
            except Exception: pass
        import gc; gc.collect(); torch.cuda.empty_cache()
        
    batch_mappings = [(i, r['T'].argmax(dim=1).cpu().numpy(), r['target_indices']) for i, r in enumerate(batch_results)]

    if debug_plots_path:
        plot_convergence(convergence_data=convergence_data, save_dir=os.path.join(debug_plots_path, 'convergence'))

    # Final transformation
    with torch.no_grad():
        model.eval()
        a_hat_cpu = apply_sb_sde_solver(model, features_a, source_to_context, n_steps=20, sigma=0.2).detach().cpu().numpy()

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
            feature_mean=None, 
            feature_std=None, 
            gene_names=adata_a.var_names.tolist()
        )

    return model, np.full(n_a, -1, dtype=np.int64), concat_adata, T_full, None, None