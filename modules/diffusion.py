import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Optional
from .belief_propagation import run_BP_diffusion
from .bp_torch import run_BP_diffusion_torch


def noise_like_crossdevice(x, seed=0):
    """Generate reproducible Gaussian noise on CPU and move to x's device.

    This ensures identical noise regardless of the target device (CPU vs CUDA),
    which is useful for deterministic backward passes.

    Args:
        x (torch.Tensor): reference tensor whose shape, dtype, and device are matched.
        seed (int): manual seed for the CPU generator.

    Returns:
        torch.Tensor: noise tensor on the same device as x.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    n = torch.randn(x.shape, dtype=x.dtype, device='cpu', generator=gen)
    return n.to(x.device)


# Continuous diffusion process

def forward_process_mean_std(x0 : torch.Tensor, t: int, alphas: Optional[torch.Tensor] = None, alpha_bars: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation of the forward process at time t.
    See Eq. (2.11) in https://arxiv.org/pdf/2403.18103
    
    Args:
        x0 (torch.Tensor): Initial sample.
        t (int): Time step.
        alphas (torch.Tensor, optional): Schedule of alpha values.
        alpha_bars (torch.Tensor, optional): Schedule of alpha_bar values.
    
    Returns:
        tuple: (mean, std) of the forward process.
    """
    if t < 1:
        raise ValueError("t must be greater than or equal to 1. t = 0 does not inject noise.")
    
    if alphas is not None:
        if alpha_bars is not None:
            raise ValueError("Either the alphas or the alpha_bars should be specified, not both.")
        else:
            # Compute the cumulative product over the first t alphas and use the final value.
            cumprod = torch.cumprod(alphas[:t], dim=0)[-1]
            mean = torch.sqrt(cumprod) * x0
            std = torch.sqrt(1.0 - cumprod)
    elif alpha_bars is not None:
        mean = torch.sqrt(alpha_bars[t - 1]) * x0
        std = torch.sqrt(1.0 - alpha_bars[t - 1])
    else:
        raise ValueError("Either the alphas or the alpha_bars should be specified.")
    return mean, std


def forward_process(x0: torch.Tensor, t: int, alphas: Optional[torch.Tensor] = None, alpha_bars: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the forward process at time t.
    See Eq. (2.11) in https://arxiv.org/pdf/2403.18103
    
    Args:
        x0 (torch.Tensor): Initial sample.
        t (int): Time step.
        alphas (torch.Tensor, optional): Schedule of alpha values.
        alpha_bars (torch.Tensor, optional): Schedule of alpha_bar values.
    
    Returns:
        torch.Tensor: Noised sample at time t.
    """
    noise = torch.randn_like(x0)
    mean, std = forward_process_mean_std(x0, t, alphas=alphas, alpha_bars=alpha_bars)
    xt = mean + std * noise       
    return xt


def posterior_process(x0: torch.Tensor, x_t: torch.Tensor, t: int, alphas: torch.Tensor, alpha_bars: torch.Tensor, deterministic: bool = False, fix_noise: bool = False) -> torch.Tensor:
    """
    Evaluate the posterior p(x_{t-1}|x_t, x_0) at time t-1 for continuous tokens.
    See Eqs (2.25) and (2.26) in  https://arxiv.org/pdf/2403.18103
    
    Args:
        x0 (torch.Tensor): Initial sample.
        x_t (torch.Tensor): Sample at time t (observation).
        t (int): Time step.
        alphas (torch.Tensor): Schedule of alpha values.
        alpha_bars (torch.Tensor): Schedule of alpha_bar values.
        deterministic (bool): If True, use the mean of the posterior.
    
    Returns:
        torch.Tensor: Sample of x_{t-1} from the posterior.
    """
    # When this function is called, t is already decremented by 1, so hereafter t - 1 is actually t - 2
    abar_t = alpha_bars[t]
    alpha_t = alphas[t]
    if t > 0:
        abar_tminus1 = alpha_bars[t - 1] # for t = 1, this is alpha_bar[0], corresponding to the first noise step
    else:
        abar_tminus1 = torch.tensor(1.0, device=x0.device, dtype=alpha_bars.dtype) # for t = 0, this is alpha_bar[-1], which does not exist. We set it to 1.0, meaning no noise is injected
    
    # notice that for t = 0, the posterior mean is equal to x0, and the posterior variance is 0
    posterior_mean = (torch.sqrt(abar_tminus1) * (1 - alpha_t) / (1 - abar_t)) * x0 \
                   + (torch.sqrt(alpha_t) * (1 - abar_tminus1) / (1 - abar_t)) * x_t
    if not deterministic:
        if fix_noise:
            noise = noise_like_crossdevice(x0, int(t))
        else:
            noise = torch.randn_like(x0)
        posterior_variance = (1 - alpha_t) * (1 - abar_tminus1) / (1 - abar_t)
        posterior_std = torch.sqrt(posterior_variance)
        x_tminus1 = posterior_mean + posterior_std * noise
    else:
        x_tminus1 = posterior_mean
    return x_tminus1


# General routine for backward process

@torch.inference_mode()
def backward_process(xt: torch.Tensor, t: int, t_final: int, model: torch.nn.Module, device: torch.device,
                     alphas: torch.Tensor, alpha_bars: torch.Tensor, clamp: bool = False, temperature: Optional[float] = None, deterministic: bool = False,
                     fix_noise: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Compute the backward process at time t.
    
    Args:
        xt (torch.Tensor): Sample at time t.
        t (int): Time step.
        t_final (int): Number of diffusion steps in the schedule.
        model (torch.nn.Module): Denoising model.
        device (torch.device): Device for computation.
        alphas (torch.Tensor): Schedule of alpha values.
        alpha_bars (torch.Tensor): Schedule of alpha_bar values.
        clamp (bool): If True, clamp the output to the most likely value.
        temperature (float, optional): Temperature for the softmax.
        deterministic (bool): If True, use the posterior mean only.
        fix_noise (bool, optional): fix noise in backward pass.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x_{t-1} sample, predicted x_0).
    """
    model.eval()
    with torch.amp.autocast('cuda'):
        t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
        x0_hat = model(xt, t_tensor)
    
    if not clamp:
        if temperature is not None:
            x0_hat = torch.nn.functional.softmax(x0_hat.float() / temperature, dim=-1)
        else:
            x0_hat = torch.nn.functional.softmax(x0_hat.float(), dim=-1)
    else:
        # Clamp to the most likely token
        print('Clamping output...', flush=True)
        x0_hat = torch.nn.functional.one_hot(torch.argmax(x0_hat.float(), dim=-1),
                                               num_classes=model.vocab_size).float()
    
    xtminus1_hat = posterior_process(x0_hat, xt, t - 1, alphas=alphas,
                                      alpha_bars=alpha_bars, deterministic=deterministic, fix_noise=fix_noise)
    return xtminus1_hat, x0_hat


def backward_process_gt(xt: torch.Tensor, t: int, bp_params: tuple, device: torch.device,
                        alphas: torch.Tensor = None, alpha_bars: torch.Tensor = None,
                        deterministic: bool = False, factorized_layers: int = 0, fix_noise: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the backward process at time t using ground-truth via belief propagation (BP).
    
    Args:
        xt (torch.Tensor): Sample at time t.
        t (int): Time step.
        bp_params (tuple): Parameters (M, l, q) for the BP computation.
        device (torch.device): Device for computation.
        alphas (torch.Tensor): Schedule of alpha values.
        alpha_bars (torch.Tensor): Schedule of alpha_bar values.
        deterministic (bool): If True, use the mean of the posterior.
        factorized_layers (int): number of tree levels to factorize in BP_k.
        fix_noise (bool, optional): fix noise in backward pass.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x_{t-1} sample, predicted x_0 from BP marginals).
    """
    M, l, q = bp_params
    x0_hat = np.zeros((xt.shape[0], xt.shape[1], xt.shape[2]), dtype=np.float32)
    field_intensity = np.sqrt(alpha_bars[t - 1]) / (1 - alpha_bars[t - 1])
    
    for i in range(xt.shape[0]):
        field = xt[i] * field_intensity
        temp = run_BP_diffusion(M, l, q, field, factorized_layers=factorized_layers)
        x0_hat[i] = temp
    
    x0_hat = torch.tensor(x0_hat)
    alpha_bars = torch.from_numpy(alpha_bars)
    xt = torch.from_numpy(xt)
    alphas = torch.from_numpy(alphas)
    xtminus1_hat = posterior_process(x0_hat, xt, t - 1, alphas=alphas,
                                      alpha_bars=alpha_bars, deterministic=deterministic, fix_noise=fix_noise)
    return xtminus1_hat, x0_hat


@torch.inference_mode()
def backward_process_gt_torch(xt: torch.Tensor, t: int, bp_params: tuple, device: torch.device,
                              alphas: torch.Tensor = None, alpha_bars: torch.Tensor = None,
                              deterministic: bool = False, factorized_layers: int = 0, 
                              fix_noise: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the backward process at time t using ground-truth BP (batched torch version).
    
    This is a drop-in replacement for backward_process_gt that uses the torch-based BP
    implementation for better GPU utilization and batched processing.
    
    Args:
        xt (torch.Tensor): Sample at time t, shape (B, L, Q).
        t (int): Time step.
        bp_params (tuple): Parameters (M, l, q) for the BP computation.
        device (torch.device): Device for computation.
        alphas (torch.Tensor): Schedule of alpha values.
        alpha_bars (torch.Tensor): Schedule of alpha_bar values.
        deterministic (bool): If True, use the mean of the posterior.
        factorized_layers (int): Number of factorized layers (must be 0 for torch version).
        fix_noise (bool, optional): Fix noise in backward pass.
    
    Returns:
        xtminus1_hat (torch.Tensor): Sample at time t-1.
        x0_hat (torch.Tensor): Predicted x0 from BP marginals.
    """
    M, l, q = bp_params
    
    # Convert inputs to tensors if needed
    if isinstance(xt, np.ndarray):
        xt = torch.from_numpy(xt).float()
    if isinstance(alpha_bars, np.ndarray):
        alpha_bars = torch.from_numpy(alpha_bars).float()
    if isinstance(alphas, np.ndarray):
        alphas = torch.from_numpy(alphas).float()
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).float()
    
    # Move to device
    xt = xt.to(device)
    alpha_bars = alpha_bars.to(device)
    alphas = alphas.to(device)
    M = M.to(device)
    
    # Compute field intensity
    field_intensity = torch.sqrt(alpha_bars[t - 1]) / (1 - alpha_bars[t - 1])
    
    # Compute field: (B, L, Q) * scalar
    field = xt * field_intensity
    
    # Run batched BP
    x0_hat = run_BP_diffusion_torch(M, l, q, field, factorized_layers=factorized_layers)
    
    # Compute posterior sample
    xtminus1_hat = posterior_process(x0_hat, xt, t - 1, alphas=alphas,
                                     alpha_bars=alpha_bars, deterministic=deterministic, fix_noise=fix_noise)
    
    return xtminus1_hat, x0_hat


@torch.inference_mode()
def backward_process_prior(xt: torch.Tensor, t: int, device: torch.device,
                          alphas: torch.Tensor = None, alpha_bars: torch.Tensor = None,
                          deterministic: bool = False, fix_noise: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the backward process at time t using a prior.

    Args:
        xt (torch.Tensor): Sample at time t.
        t (int): Time step.
        device (torch.device): Device for computation.
        alphas (torch.Tensor): Schedule of alpha values.
        alpha_bars (torch.Tensor): Schedule of alpha_bar values.
        deterministic (bool): If True, use the mean of the posterior.
        fix_noise (bool, optional): fix noise in backward pass.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x_{t-1} sample, predicted x_0 from prior).
    """
    field_intensity = torch.sqrt(alpha_bars[t - 1]) / (1 - alpha_bars[t - 1])
    x0_hat = (xt * field_intensity).softmax(dim=-1)
    xtminus1_hat = posterior_process(x0_hat, xt, t - 1, alphas=alphas,
                                      alpha_bars=alpha_bars, deterministic=deterministic, fix_noise=fix_noise)
    return xtminus1_hat, x0_hat


@torch.inference_mode()
def backward_process_empirical(xt: torch.Tensor, x0s: torch.Tensor, t: int, device: torch.device,
                               alphas: torch.Tensor = None, alpha_bars: torch.Tensor = None,
                               deterministic: bool = False, fix_noise: Optional[bool] = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the backward process at time t using an empirical distribution.

    Args:
        xt (torch.Tensor): Sample at time t.
        x0s (torch.Tensor): Samples at time 0.
        t (int): Time step.
        device (torch.device): Device for computation.
        alphas (torch.Tensor): Schedule of alpha values.
        alpha_bars (torch.Tensor): Schedule of alpha_bar values.
        deterministic (bool): If True, use the mean of the posterior.
        fix_noise (bool, optional): fix noise in backward pass.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x_{t-1} sample, predicted x_0 from empirical
            distribution).
    """
    scaled_means = torch.sqrt(alpha_bars[t - 1]) * x0s
    var = 1 - alpha_bars[t - 1]
    # (B, P): squared L2 distance from each sample to each scaled training point
    sq_diff = torch.sum((scaled_means[None, :, :, :] - xt[:, None, :, :]) ** 2, dim=(2, 3))
    weights = torch.nn.functional.softmax(-0.5 * sq_diff / var, dim=1)  # (B, P)
    x0_hat = torch.einsum('bp,plq->blq', weights, x0s)
    xtminus1_hat = posterior_process(x0_hat, xt, t - 1, alphas=alphas,
                                      alpha_bars=alpha_bars, deterministic=deterministic, fix_noise=fix_noise)
    return xtminus1_hat, x0_hat


############################################################################
# General routine for forward process noising
############################################################################

@torch.inference_mode()
def generate_noisy_sequences_single(x0: torch.Tensor, *, t_final: int, alpha_bars: torch.Tensor, device: torch.device, reweight: bool = False, single_step: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create ONE noised version `xt` of an input token / sequence `x0`.

    Args:
        x0 (torch.Tensor): original sample, shape [..., vocab] or [..., feat].
        t_final (int): maximum diffusion step (inclusive).
        alpha_bars (torch.Tensor): pre-computed schedule, shape [t_final] on
            ``device``.
        device (torch.device): target device (same as x0 for best performance).
        reweight (bool): if True, sample timesteps from a Gaussian centered at
            t/T = 0.15 (sigma = 0.15) to up-weight the critical time region.
            Defaults to False (uniform sampling). The Gaussian option is kept
            for legacy experiments but is not used in the paper results.
        single_step (int | None): force a fixed timestep (useful for evaluation).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(xt, x0, t)`` where
            *xt* is the noised sample, *x0* the original, and *t* the chosen
            timestep (scalar).
    """
    if single_step is not None:
        if not (1 <= single_step <= t_final):
            raise ValueError("single_step must be in [1, t_final]")
        t = torch.tensor(single_step, device=device, dtype=torch.long)
    else:
        if reweight:
            timesteps = torch.arange(1, t_final + 1, device=device, dtype=torch.float32)
            s = timesteps / float(t_final)  # in (0, 1]

            # Gaussian parameters in normalized time (peaks near critical time)
            center = 0.15   # peak near the critical diffusion time
            sigma = 0.15    # spread; covers roughly [0, 0.5]

            # Unnormalized weights: Gaussian in s
            weights = torch.exp(-0.5 * ((s - center) / sigma) ** 2)

            # Optional: give a very small floor so extremes are never exactly 0
            weights = weights + 1e-8

            # Normalize to get probabilities
            probs = weights / weights.sum()

            # Sample an index according to probs
            idx = torch.multinomial(probs, 1)  # shape [1], values in [0, t_final-1]
            t = timesteps[idx].long().squeeze(0)  # convert back to 1..t_final scalar
        else:
            t = torch.randint(1, t_final + 1, (), device=device)
    xt = forward_process(x0, t, alpha_bars=alpha_bars)
    return xt, x0, t