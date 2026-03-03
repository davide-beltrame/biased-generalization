import torch
import numpy as np


def alpha_bars_schedule(t_final: int, device: torch.device, s: float = 1e-4, schedule: str = 'linear'):
    """Generate a schedule of cumulative product alpha_bar values.

    The ``'linear'`` schedule follows the Diffusion-LM convention
    (Li et al., 2022) which rescales the base betas by ``1000 / t_final``
    so that the total noise level is approximately invariant to T::

        scale = 1000 / t_final
        beta_start = scale * 1e-4
        beta_end   = scale * 0.02
 
    After rescaling by 1000 / t_final = 2, 
    the actual per-step betas are [2e-4, 0.04].

    Args:
        t_final (int): number of diffusion steps T.
        device (torch.device): target device.
        s (float): offset for the sqrt schedule (only used when
            schedule='sqrt').
        schedule (str): ``'linear'`` (default, Diffusion-LM) or ``'sqrt'``
            (not recommended, alpha is not monotonically decreasing).

    Returns:
        torch.Tensor: alpha_bar values of shape ``[t_final]``.
    """
    if schedule == 'sqrt':
        # from Appendix A: https://arxiv.org/pdf/2205.14217
        alpha_bar = lambda t: 1-np.sqrt(t + s)
        return alpha_bar(torch.linspace(0, 1, t_final).float()).to(device=device)
    elif schedule == 'linear':
        # from Diffusion-LM: https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/gaussian_diffusion.py    
        scale = 1000 / t_final
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        beta = torch.linspace(beta_start, beta_end, t_final, dtype=torch.float64)
        alpha = 1 - beta
        return torch.cumprod(alpha, dim=0).to(device=device)
    

def get_alpha_beta_from_alpha_bar(alpha_bar: torch.Tensor):
    """
    Compute the alpha and beta schedules from the alpha_bar schedule.
    Args:
        alpha_bar (torch.Tensor): Schedule of alpha_bar values.
    Returns:
        torch.Tensor: Schedule of alpha values.
        torch.Tensor: Schedule of beta values.
    """
    if alpha_bar.dim() != 1:
        raise ValueError("alpha_bar should be a 1-D tensor.")
    alpha_prev = torch.cat([torch.ones(1, device=alpha_bar.device), alpha_bar[:-1]])
    alpha = alpha_bar / alpha_prev
    beta = 1 - alpha
    return alpha, beta
