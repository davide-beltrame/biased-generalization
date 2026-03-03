import torch


def compute_mse_loss(x0s, x0_hats, rescale_coeffs=None):
    """Compute per-sample MSE reconstruction loss for the denoising transformer.

    Args:
        x0s (torch.Tensor): ground-truth starting points (one-hot), shape
            ``(B, L, Q)``.
        x0_hats (torch.Tensor): predicted starting points, same shape.
        rescale_coeffs (torch.Tensor, optional): per-sample VLB coefficients.
            If given, each sample's MSE is multiplied by its coefficient
            (corresponds to L_simple, Eq. 14 in the DDPM paper).

    Returns:
        torch.Tensor: per-sample loss of shape ``(B,)``.
    """
    if rescale_coeffs is None:
        return ((x0s - x0_hats) ** 2).mean(dim=list(range(1, len(x0s.shape)))) 
    else:
        # NOTE: this corresponds to L_simple (Eq. 14) in DDPM https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
        return ((x0s - x0_hats) ** 2).mean(dim=list(range(1, len(x0s.shape)))) * rescale_coeffs


def compute_cross_entropy_loss(x0s, x0_hats, rescale_coeffs=None):
    """Compute per-token cross-entropy loss between labels and predictions.

    Args:
        x0s (torch.Tensor): integer class labels, shape ``(B, L)``.
        x0_hats (torch.Tensor): logits, shape ``(B, L, Q)``.
        rescale_coeffs (torch.Tensor, optional): per-sample VLB coefficients.

    Returns:
        torch.Tensor: per-token loss, shape ``(B * L,)`` (or
        ``(B * L,)`` element-wise multiplied by rescale coefficients).
    """
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
    if rescale_coeffs is None:
        return cross_entropy_loss(
            torch.flatten(x0_hats, start_dim=0, end_dim=1),  # Flatten the tensor (we pool all token embeddings from all elements in the batch).
            torch.flatten(x0s.to(dtype=torch.int64), start_dim=0, end_dim=1),  # Flatten the tensor (we pool all token embeddings from all elements in the batch).
        )
    else:
        loss = cross_entropy_loss(
            torch.flatten(x0_hats, start_dim=0, end_dim=1),  # Flatten the tensor (we pool all token embeddings from all elements in the batch).
            torch.flatten(x0s.to(dtype=torch.int64), start_dim=0, end_dim=1),  # Flatten the tensor (we pool all token embeddings from all elements in the batch).
        )
        rescale_coeffs = rescale_coeffs.unsqueeze(1).repeat(1,x0_hats.shape[1])
        return loss * rescale_coeffs.flatten()


def get_loss_coefficients_continuous_new(alpha_bars, alphas, timesteps_batch):
    """Compute normalised VLB coefficients for x_0-prediction at given timesteps.

    The coefficients are normalised by their mean to maintain training stability.
    This preserves the relative U-shaped weighting (high at small t, low at large t)
    while preventing gradient explosion from the extremely large raw coefficient
    values.

    Args:
        alpha_bars (torch.Tensor): cumulative product schedule, length ``T``.
        alphas (torch.Tensor): per-step alpha values, length ``T``.
        timesteps_batch (torch.Tensor): timesteps for the current batch,
            values in ``{1, ..., T}``.

    Returns:
        torch.Tensor: per-sample weight of shape ``(B,)``.
    """
    device = alpha_bars.device
    T = len(alpha_bars)
    
    # 1. Compute alpha_bar_{t-1} for all t and prepend 1.0 for t=0
    alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]])
    
    # 2. Compute posterior variance (sigma_t^2) for all t as beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    betas = 1 - alphas
    sigma_t_squared = betas * (1 - alpha_bar_prev) / (1 - alpha_bars)
    
    # 3. The true variance at t=1 is 0. We clip it to the value at t=2 (index 1) to avoid division by zero
    sigma_t_squared_clipped = sigma_t_squared.clone()
    if len(sigma_t_squared) > 1:
        sigma_t_squared_clipped[0] = sigma_t_squared[1]
    
    # 4. Compute full array of coefficients as lambda_t = (beta_t^2 * alpha_bar_{t-1}) / (2 * sigma_t^2 * (1 - alpha_bar_t)^2)
    numerator = (betas ** 2) * alpha_bar_prev
    denominator = 2 * sigma_t_squared_clipped * ((1 - alpha_bars) ** 2)
    
    all_coeffs = numerator / denominator
    
    # 5. Normalize by mean to prevent gradient explosion
    all_coeffs = all_coeffs / all_coeffs.mean()
    all_coeffs = torch.clamp(all_coeffs, min=0.01, max=50.0)  # Prevent extreme outliers.
    
    # 6. Gather specific weights for the batch (timesteps_batch contains t in [1, ..., T]. We need indices [0, ..., T-1].)
    batch_indices = timesteps_batch - 1
    return all_coeffs[batch_indices]
