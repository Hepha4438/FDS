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
    Train global transformation model (Đã gỡ bỏ hoàn toàn Entropy).
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

    logging.info("  [M-step] Chạy Adam Optimizer...")
    for step in range(steps_per_iter):
        if hasattr(model, 'update_temperature'):
            model.update_temperature(current_step=step, max_steps=steps_per_iter)
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

        # ===== Combined loss (Thuần túy, không có hàm phạt) =====
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