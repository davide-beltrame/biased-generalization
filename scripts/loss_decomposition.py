"""
loss_decomposition.py - Loss decomposition: exact distillation vs excess data-dependent term
Reproduces Fig. 4(b).
================================================================================

Decomposes the cross-entropy denoising loss into two terms to reveal the
coexistence of generalization and memorization in the biased generalization
phase (Sec. 4.3, Eq. 4 in the paper):

  l_t(theta, x_0, x_t) = l*_t (exact distillation) + l~_t (excess data-dependent)

where:
- l*_t = - x_hat_0(x_t)^T log x_hat_0^theta(x_t):
  cross-entropy between model predictions and the exact BP posterior.
- l~_t = - (x_0 - x_hat_0(x_t))^T log x_hat_0^theta(x_t):
  excess term measuring data-dependent overconfidence.

On test data, E[l~_t] = 0 by construction for any fixed theta. On training
data, the excess term is finite and can be optimized. In the biased
generalization phase, *both* terms decrease simultaneously on training data,
illustrating coexistence of generalization and memorization.

Modes:
1. Single-seed mode: evaluate one model directory.
2. Multi-seed mode: aggregate across multiple seeds with mean +/- std.

Usage (run from scripts/ directory)::

    # multi-seed loss decomposition (Fig. 4(b), n=12k, 15 seeds)
    python loss_decomposition.py \\
        --output-root ../plots/ \\
        --base-dir "<base>/results_transformer_pick_0_for_training" \\
        --seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14" \\
        --subpath "test_model_script_30000_12000_cross_entropy_loss/8_512" \\
        --batch-size 128 --seed 0 \\
        --fixed-t-ratio 0.3 --plots

    Epoch selection: Fig. 4(b) scans all checkpoints. Omit --epochs to let
    the script discover every available checkpoint and plot the full
    trajectory. The NN-divergence and test-loss minima are then visible as
    vertical dashed lines on the output plot.
"""

import sys
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
import matplotlib.colors as mcolors

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.noise_schedules import alpha_bars_schedule
from modules.losses import compute_cross_entropy_loss
from modules.bp_torch import run_BP_diffusion_torch

from utils import (
    load_params,
    load_data,
    create_model,
    load_checkpoint,
    setup_device,
    find_available_epochs,
)

# Optional import for epoch-selection heuristic (used in --select mode)
try:
    from nn_divergence_selector import rank_candidates, collect_exp2_data_for_plot, plot_exp2_diagnostic
    HAS_NN_SELECTOR = True
except ImportError:
    HAS_NN_SELECTOR = False

# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class LossDecompositionResults:
    """Container for per-epoch loss decomposition results.

    Stores both fixed-timestep and mean-over-timesteps modes, each split
    into total / BP / memorisation / generalisation components.
    """
    epochs: List[int] = field(default_factory=list)
    
    # ========== FIXED TIMESTEP MODE ==========
    # Train metrics (fixed t)
    train_loss_total_fixed: List[float] = field(default_factory=list)
    train_loss_p_star_fixed: List[float] = field(default_factory=list)
    train_loss_residual_fixed: List[float] = field(default_factory=list)
    
    # Test metrics (fixed t)
    test_loss_total_fixed: List[float] = field(default_factory=list)
    test_loss_p_star_fixed: List[float] = field(default_factory=list)
    test_loss_residual_fixed: List[float] = field(default_factory=list)
    
    # ========== EXPECTATION (MEAN) MODE ==========
    # Train metrics (averaged over all timesteps)
    train_loss_total_mean: List[float] = field(default_factory=list)
    train_loss_p_star_mean: List[float] = field(default_factory=list)
    train_loss_residual_mean: List[float] = field(default_factory=list)
    
    # Test metrics (averaged over all timesteps)
    test_loss_total_mean: List[float] = field(default_factory=list)
    test_loss_p_star_mean: List[float] = field(default_factory=list)
    test_loss_residual_mean: List[float] = field(default_factory=list)
    
    # Metadata
    fixed_timestep: Optional[int] = None
    t_final: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'epochs': self.epochs,
            'fixed_timestep': self.fixed_timestep,
            't_final': self.t_final,
            # Fixed mode
            'train_loss_total_fixed': self.train_loss_total_fixed,
            'train_loss_p_star_fixed': self.train_loss_p_star_fixed,
            'train_loss_residual_fixed': self.train_loss_residual_fixed,
            'test_loss_total_fixed': self.test_loss_total_fixed,
            'test_loss_p_star_fixed': self.test_loss_p_star_fixed,
            'test_loss_residual_fixed': self.test_loss_residual_fixed,
            # Mean mode
            'train_loss_total_mean': self.train_loss_total_mean,
            'train_loss_p_star_mean': self.train_loss_p_star_mean,
            'train_loss_residual_mean': self.train_loss_residual_mean,
            'test_loss_total_mean': self.test_loss_total_mean,
            'test_loss_p_star_mean': self.test_loss_p_star_mean,
            'test_loss_residual_mean': self.test_loss_residual_mean,
        }


# ============================================================================
# Core Loss Computation Functions
# ============================================================================

def compute_bp_soft_targets(
    xt: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    rho: int,
    k: int,
    vocab_size: int,
) -> torch.Tensor:
    """
    Compute BP (Belief Propagation) soft targets for given noisy inputs.
    
    Args:
        xt: Noisy input (B, N, Q) in one-hot space
        t: Timesteps (B,)
        alpha_bars: Noise schedule
        rho, k, vocab_size: BP parameters
        
    Returns:
        BP soft targets (B, N, Q) as probability distributions
    """
    B = xt.shape[0]
    device = xt.device
    
    # Get alpha_bar for each sample
    alpha_bars_t = alpha_bars[t - 1]  # (B,)
    
    # Compute field intensity: sqrt(ab) / (1 - ab)
    field_intensity = torch.sqrt(alpha_bars_t) / (1 - alpha_bars_t + 1e-8)  # (B,)
    
    # Apply field: scale xt by field intensity
    field = xt * field_intensity[:, None, None]  # (B, N, Q)
    
    # Run BP to get soft posterior
    bp_targets = run_BP_diffusion_torch(rho, k, vocab_size, field, factorized_layers=0)
    
    return bp_targets


def compute_loss_decomposition_batch(
    model: nn.Module,
    x0_oh: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    rho: int,
    k: int,
    vocab_size: int,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    """
    Compute loss decomposition for a batch.
    
    Args:
        model: The neural network model
        x0_oh: One-hot encoded clean data (B, L, V)
        t: Timesteps (B,)
        alpha_bars: Noise schedule
        rho, k, vocab_size: BP parameters
        noise: Pre-generated noise (B, L, V). If None, generates fresh noise.
    
    Returns:
        (loss_total, loss_p_star, loss_residual) averaged over batch
    """
    device = x0_oh.device
    B = x0_oh.shape[0]
    
    # Forward diffusion: add noise
    alpha_bar_t = alpha_bars[t - 1][:, None, None]
    if noise is None:
        noise = torch.randn_like(x0_oh)
    xt = torch.sqrt(alpha_bar_t) * x0_oh + torch.sqrt(1 - alpha_bar_t) * noise
    
    # Get model predictions (logits)
    with torch.no_grad():
        logits = model(xt.float(), t)
    
    # --- Loss Total: CE against hard labels (one-hot data) ---
    targets_hard = torch.argmax(x0_oh, dim=-1).long()  # (B, N)
    loss_total_per_token = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),  # (B*N, Q)
        targets_hard.view(-1),  # (B*N,)
        reduction='none'
    )  # (B*N,)
    loss_total = loss_total_per_token.mean().item()
    
    # --- Loss p* (Distillation): CE where target=BP soft posterior ---
    # Get BP soft targets
    bp_targets = compute_bp_soft_targets(xt, t, alpha_bars, rho, k, vocab_size)
    
    # Cross-entropy with soft targets: -sum(p_BP * log(p_model))
    log_probs = nn.functional.log_softmax(logits, dim=-1)  # (B, N, Q)
    loss_p_star_per_token = -(bp_targets * log_probs).sum(dim=-1)  # (B, N)
    loss_p_star = loss_p_star_per_token.mean().item()
    
    # --- Loss Residual ---
    loss_residual = loss_total - loss_p_star
    
    return loss_total, loss_p_star, loss_residual


def evaluate_epoch(
    model: nn.Module,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    alpha_bars: torch.Tensor,
    rho: int,
    k: int,
    vocab_size: int,
    t_final: int,
    device: torch.device,
    model_dir: Path,
    fixed_timestep: int,
    noise_maps: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate loss decomposition for a single epoch in BOTH modes:
    - Fixed timestep mode (single t for cleanup phase analysis)
    - Mean mode (expectation over all timesteps for global dynamics)
    
    Uses pre-generated noise maps to ensure reproducibility across epochs.
    
    Args:
        noise_maps: Dict with pre-generated noise tensors:
            - 'train_noise_fixed': Noise for train set, fixed t mode
            - 'train_noise_mean': Noise for train set, mean t mode
            - 'train_t_mean': Random timesteps for train set, mean t mode
            - 'test_noise_fixed': Noise for test set, fixed t mode
            - 'test_noise_mean': Noise for test set, mean t mode
            - 'test_t_mean': Random timesteps for test set, mean t mode
    
    Returns:
        (train_metrics, test_metrics) dictionaries with both modes
    """
    # Load checkpoint
    ckpt_path = model_dir / f"test_model_script_epoch_{epoch}.pt"
    if not load_checkpoint(model, str(ckpt_path), device=device, training=False):
        logger.warning(f"Skipping epoch {epoch}: Checkpoint not found")
        return None, None
    
    model.eval()
    
    batch_size = train_loader.batch_size
    
    # ===== FIXED TIMESTEP MODE =====
    t_fixed_tensor = torch.tensor([fixed_timestep], device=device)
    
    # Train - Fixed
    train_total_fixed, train_p_star_fixed, train_count = 0.0, 0.0, 0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        B = batch.shape[0]
        t = t_fixed_tensor.expand(B)  # All samples at same timestep
        
        # Get pre-generated noise for this batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + B
        noise = noise_maps['train_noise_fixed'][start_idx:end_idx]
        
        l_total, l_p_star, _ = compute_loss_decomposition_batch(
            model, batch, t, alpha_bars, rho, k, vocab_size, noise=noise
        )
        train_total_fixed += l_total * B
        train_p_star_fixed += l_p_star * B
        train_count += B
    
    # Test - Fixed
    test_total_fixed, test_p_star_fixed, test_count = 0.0, 0.0, 0
    for batch_idx, batch in enumerate(test_loader):
        batch = batch.to(device)
        B = batch.shape[0]
        t = t_fixed_tensor.expand(B)
        
        start_idx = batch_idx * batch_size
        end_idx = start_idx + B
        noise = noise_maps['test_noise_fixed'][start_idx:end_idx]
        
        l_total, l_p_star, _ = compute_loss_decomposition_batch(
            model, batch, t, alpha_bars, rho, k, vocab_size, noise=noise
        )
        test_total_fixed += l_total * B
        test_p_star_fixed += l_p_star * B
        test_count += B
    
    # ===== MEAN (EXPECTATION) MODE =====
    # Train - Mean over pre-generated random timesteps
    train_total_mean, train_p_star_mean = 0.0, 0.0
    train_count_mean = 0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        B = batch.shape[0]
        
        start_idx = batch_idx * batch_size
        end_idx = start_idx + B
        t = noise_maps['train_t_mean'][start_idx:end_idx]
        noise = noise_maps['train_noise_mean'][start_idx:end_idx]
        
        l_total, l_p_star, _ = compute_loss_decomposition_batch(
            model, batch, t, alpha_bars, rho, k, vocab_size, noise=noise
        )
        train_total_mean += l_total * B
        train_p_star_mean += l_p_star * B
        train_count_mean += B
    
    # Test - Mean over pre-generated random timesteps
    test_total_mean, test_p_star_mean = 0.0, 0.0
    test_count_mean = 0
    for batch_idx, batch in enumerate(test_loader):
        batch = batch.to(device)
        B = batch.shape[0]
        
        start_idx = batch_idx * batch_size
        end_idx = start_idx + B
        t = noise_maps['test_t_mean'][start_idx:end_idx]
        noise = noise_maps['test_noise_mean'][start_idx:end_idx]
        
        l_total, l_p_star, _ = compute_loss_decomposition_batch(
            model, batch, t, alpha_bars, rho, k, vocab_size, noise=noise
        )
        test_total_mean += l_total * B
        test_p_star_mean += l_p_star * B
        test_count_mean += B
    
    # Compile metrics
    train_metrics = {
        # Fixed mode
        'loss_total_fixed': train_total_fixed / train_count,
        'loss_p_star_fixed': train_p_star_fixed / train_count,
        'loss_residual_fixed': (train_total_fixed - train_p_star_fixed) / train_count,
        # Mean mode
        'loss_total_mean': train_total_mean / train_count_mean,
        'loss_p_star_mean': train_p_star_mean / train_count_mean,
        'loss_residual_mean': (train_total_mean - train_p_star_mean) / train_count_mean,
    }
    
    test_metrics = {
        # Fixed mode
        'loss_total_fixed': test_total_fixed / test_count,
        'loss_p_star_fixed': test_p_star_fixed / test_count,
        'loss_residual_fixed': (test_total_fixed - test_p_star_fixed) / test_count,
        # Mean mode
        'loss_total_mean': test_total_mean / test_count_mean,
        'loss_p_star_mean': test_p_star_mean / test_count_mean,
        'loss_residual_mean': (test_total_mean - test_p_star_mean) / test_count_mean,
    }
    
    return train_metrics, test_metrics


# ============================================================================
# Plotting Functions
# ============================================================================

def add_min_vline(ax, epochs: np.ndarray, values: np.ndarray, color: str, 
                  linestyle: str = '--', alpha: float = 0.6, label: str = None):
    """
    Add a vertical line at the epoch where the loss curve reaches its minimum.
    
    Args:
        ax: Matplotlib axis to draw on
        epochs: Array of epoch values (x-axis)
        values: Array of loss values (y-axis)
        color: Color for the vertical line
        linestyle: Line style (default '--')
        alpha: Transparency (default 0.6)
        label: Optional label for legend
    
    Returns:
        min_epoch: The epoch at which minimum occurs
    """
    values = np.array(values)
    if len(values) == 0:
        return None
    min_idx = np.argmin(values)
    min_epoch = epochs[min_idx]
    min_value = values[min_idx]
    ax.axvline(x=min_epoch, color=color, linestyle=linestyle, alpha=alpha, linewidth=1.5, label=label)
    return min_epoch


def plot_loss_decomposition(
    results: LossDecompositionResults,
    save_path: Path,
    title_suffix: str = "",
):
    """
    Plot loss decomposition curves for train and test, showing BOTH modes.
    2x2 grid: rows = Fixed vs Mean, columns = Train vs Test
    
    NOTE: Uses symlog scale for residuals (can be negative)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    epochs = np.array(results.epochs)
    
    # --- Row 0: FIXED TIMESTEP MODE ---
    # Train Fixed
    ax = axes[0, 0]
    ax.plot(epochs, results.train_loss_total_fixed, 'k-', linewidth=2, label='Total Loss', marker='o', markersize=4)
    ax.plot(epochs, results.train_loss_p_star_fixed, 'b-', linewidth=2, label='p* (Distillation)', marker='s', markersize=4)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.train_loss_total_fixed, 'black')
    add_min_vline(ax, epochs, results.train_loss_p_star_fixed, 'blue')
    # Residual on secondary axis with symlog
    ax2 = ax.twinx()
    ax2.plot(epochs, results.train_loss_residual_fixed, 'r-', linewidth=2, label='Residual', marker='^', markersize=4, alpha=0.7)
    add_min_vline(ax2, epochs, results.train_loss_residual_fixed, 'red')
    ax2.set_ylabel('Residual (symlog)', color='r', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log)', fontsize=12)
    ax.set_title(f'Train - Fixed t={results.fixed_timestep}', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Test Fixed
    ax = axes[0, 1]
    ax.plot(epochs, results.test_loss_total_fixed, 'k-', linewidth=2, label='Total Loss', marker='o', markersize=4)
    ax.plot(epochs, results.test_loss_p_star_fixed, 'b-', linewidth=2, label='p* (Distillation)', marker='s', markersize=4)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.test_loss_total_fixed, 'black')
    add_min_vline(ax, epochs, results.test_loss_p_star_fixed, 'blue')
    ax2 = ax.twinx()
    ax2.plot(epochs, results.test_loss_residual_fixed, 'r-', linewidth=2, label='Residual', marker='^', markersize=4, alpha=0.7)
    add_min_vline(ax2, epochs, results.test_loss_residual_fixed, 'red')
    ax2.set_ylabel('Residual (symlog)', color='r', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log)', fontsize=12)
    ax.set_title(f'Test - Fixed t={results.fixed_timestep}', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Row 1: MEAN (EXPECTATION) MODE ---
    # Train Mean
    ax = axes[1, 0]
    ax.plot(epochs, results.train_loss_total_mean, 'k-', linewidth=2, label='Total Loss', marker='o', markersize=4)
    ax.plot(epochs, results.train_loss_p_star_mean, 'b-', linewidth=2, label='p* (Distillation)', marker='s', markersize=4)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.train_loss_total_mean, 'black')
    add_min_vline(ax, epochs, results.train_loss_p_star_mean, 'blue')
    ax2 = ax.twinx()
    ax2.plot(epochs, results.train_loss_residual_mean, 'r-', linewidth=2, label='Residual', marker='^', markersize=4, alpha=0.7)
    add_min_vline(ax2, epochs, results.train_loss_residual_mean, 'red')
    ax2.set_ylabel('Residual (symlog)', color='r', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log)', fontsize=12)
    ax.set_title(f'Train - Mean over t∈[1,{results.t_final}]', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Test Mean
    ax = axes[1, 1]
    ax.plot(epochs, results.test_loss_total_mean, 'k-', linewidth=2, label='Total Loss', marker='o', markersize=4)
    ax.plot(epochs, results.test_loss_p_star_mean, 'b-', linewidth=2, label='p* (Distillation)', marker='s', markersize=4)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.test_loss_total_mean, 'black')
    add_min_vline(ax, epochs, results.test_loss_p_star_mean, 'blue')
    ax2 = ax.twinx()
    ax2.plot(epochs, results.test_loss_residual_mean, 'r-', linewidth=2, label='Residual', marker='^', markersize=4, alpha=0.7)
    add_min_vline(ax2, epochs, results.test_loss_residual_mean, 'red')
    ax2.set_ylabel('Residual (symlog)', color='r', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log)', fontsize=12)
    ax.set_title(f'Test - Mean over t∈[1,{results.t_final}]', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Loss Decomposition Analysis{title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {save_path}")


def plot_combined_view(
    results: LossDecompositionResults,
    save_path: Path,
):
    """
    Combined view comparing Fixed vs Mean modes side by side.
    3 rows (Total, p*, Residual) x 2 cols (Fixed, Mean), with Train+Test overlaid.
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    
    epochs = np.array(results.epochs)
    
    metrics = [
        ('Total Loss', 'loss_total'),
        ('p* (Distillation)', 'loss_p_star'),
        ('Residual', 'loss_residual'),
    ]
    
    for row, (metric_name, metric_key) in enumerate(metrics):
        # Fixed mode column
        ax = axes[row, 0]
        train_vals = getattr(results, f'train_{metric_key}_fixed')
        test_vals = getattr(results, f'test_{metric_key}_fixed')
        ax.plot(epochs, train_vals, 'b-', linewidth=2, label='Train', marker='o', markersize=4)
        ax.plot(epochs, test_vals, 'r--', linewidth=2, label='Test', marker='s', markersize=4)
        # Add vertical lines at minima
        add_min_vline(ax, epochs, train_vals, 'blue')
        add_min_vline(ax, epochs, test_vals, 'red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{metric_name} - Fixed t={results.fixed_timestep}', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Mean mode column
        ax = axes[row, 1]
        train_vals = getattr(results, f'train_{metric_key}_mean')
        test_vals = getattr(results, f'test_{metric_key}_mean')
        ax.plot(epochs, train_vals, 'b-', linewidth=2, label='Train', marker='o', markersize=4)
        ax.plot(epochs, test_vals, 'r--', linewidth=2, label='Test', marker='s', markersize=4)
        # Add vertical lines at minima
        add_min_vline(ax, epochs, train_vals, 'blue')
        add_min_vline(ax, epochs, test_vals, 'red')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{metric_name} - Mean over t', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Loss Decomposition: Fixed t vs Mean t | Train vs Test', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combined plot to {save_path}")


def plot_fixed_vs_mean_overlay(
    results: LossDecompositionResults,
    save_path: Path,
):
    """
    Overlay plot comparing Fixed t vs Mean t for each metric.
    Goal: See if Phase 2 "U-turn" is sharper at fixed low timestep.
    
    2x2 grid: rows = Train vs Test, cols = p* vs Residual
    Each subplot overlays Fixed (solid) vs Mean (dashed)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = np.array(results.epochs)
    
    # Colors
    color_fixed = 'blue'
    color_mean = 'orange'
    
    # --- Row 0: Train ---
    # Train p*
    ax = axes[0, 0]
    ax.plot(epochs, results.train_loss_p_star_fixed, '-', color=color_fixed, linewidth=2.5, 
            label=f'p* Fixed (t={results.fixed_timestep})', marker='o', markersize=4)
    ax.plot(epochs, results.train_loss_p_star_mean, '--', color=color_mean, linewidth=2.5, 
            label='p* Mean', marker='s', markersize=4, alpha=0.8)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.train_loss_p_star_fixed, color_fixed)
    add_min_vline(ax, epochs, results.train_loss_p_star_mean, color_mean)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('p* Loss', fontsize=12)
    ax.set_title('Train: p* (Distillation Loss)', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Train Residual
    ax = axes[0, 1]
    ax.plot(epochs, results.train_loss_residual_fixed, '-', color=color_fixed, linewidth=2.5, 
            label=f'Residual Fixed (t={results.fixed_timestep})', marker='o', markersize=4)
    ax.plot(epochs, results.train_loss_residual_mean, '--', color=color_mean, linewidth=2.5, 
            label='Residual Mean', marker='s', markersize=4, alpha=0.8)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.train_loss_residual_fixed, color_fixed)
    add_min_vline(ax, epochs, results.train_loss_residual_mean, color_mean)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Residual (Total - p*)', fontsize=12)
    ax.set_title('Train: Residual Loss', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- Row 1: Test ---
    # Test p*
    ax = axes[1, 0]
    ax.plot(epochs, results.test_loss_p_star_fixed, '-', color=color_fixed, linewidth=2.5, 
            label=f'p* Fixed (t={results.fixed_timestep})', marker='o', markersize=4)
    ax.plot(epochs, results.test_loss_p_star_mean, '--', color=color_mean, linewidth=2.5, 
            label='p* Mean', marker='s', markersize=4, alpha=0.8)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.test_loss_p_star_fixed, color_fixed)
    add_min_vline(ax, epochs, results.test_loss_p_star_mean, color_mean)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('p* Loss', fontsize=12)
    ax.set_title('Test: p* (Distillation Loss)', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Test Residual
    ax = axes[1, 1]
    ax.plot(epochs, results.test_loss_residual_fixed, '-', color=color_fixed, linewidth=2.5, 
            label=f'Residual Fixed (t={results.fixed_timestep})', marker='o', markersize=4)
    ax.plot(epochs, results.test_loss_residual_mean, '--', color=color_mean, linewidth=2.5, 
            label='Residual Mean', marker='s', markersize=4, alpha=0.8)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, results.test_loss_residual_fixed, color_fixed)
    add_min_vline(ax, epochs, results.test_loss_residual_mean, color_mean)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Residual (Total - p*)', fontsize=12)
    ax.set_title('Test: Residual Loss', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Fixed t={results.fixed_timestep} vs Mean t Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved overlay plot to {save_path}")


def plot_multiseed_aggregation(
    all_results: Dict[int, LossDecompositionResults],
    save_path: Path,
    title_suffix: str = "",
    epoch_kl_min: Optional[float] = None,
):
    """
    Plot multi-seed aggregated results with mean ± std shading.
    
    Args:
        all_results: Dict mapping seed -> LossDecompositionResults
        save_path: Where to save the plot
    """
    seeds = sorted(all_results.keys())
    n_seeds = len(seeds)
    
    # Get epochs from first result (assume all have same epochs)
    epochs = np.array(all_results[seeds[0]].epochs)
    n_epochs = len(epochs)
    fixed_t = all_results[seeds[0]].fixed_timestep
    t_final = all_results[seeds[0]].t_final
    
    # Aggregate data into arrays: (n_seeds, n_epochs)
    metrics = {
        'train_loss_total_fixed': np.zeros((n_seeds, n_epochs)),
        'train_loss_p_star_fixed': np.zeros((n_seeds, n_epochs)),
        'train_loss_residual_fixed': np.zeros((n_seeds, n_epochs)),
        'test_loss_total_fixed': np.zeros((n_seeds, n_epochs)),
        'test_loss_p_star_fixed': np.zeros((n_seeds, n_epochs)),
        'test_loss_residual_fixed': np.zeros((n_seeds, n_epochs)),
        'train_loss_total_mean': np.zeros((n_seeds, n_epochs)),
        'train_loss_p_star_mean': np.zeros((n_seeds, n_epochs)),
        'train_loss_residual_mean': np.zeros((n_seeds, n_epochs)),
        'test_loss_total_mean': np.zeros((n_seeds, n_epochs)),
        'test_loss_p_star_mean': np.zeros((n_seeds, n_epochs)),
        'test_loss_residual_mean': np.zeros((n_seeds, n_epochs)),
    }
    
    for i, seed in enumerate(seeds):
        res = all_results[seed]
        for key in metrics:
            metrics[key][i, :] = getattr(res, key)
    
    # Compute mean and std
    means = {k: v.mean(axis=0) for k, v in metrics.items()}
    stds = {k: v.std(axis=0) for k, v in metrics.items()}
    
    # --- Plot: 2x2 grid showing p* and Residual for Fixed and Mean modes ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    def plot_with_shading(ax, epochs, mean_train, std_train, mean_test, std_test, 
                          ylabel, title, use_symlog=False):
        # Train
        ax.plot(epochs, mean_train, 'b-', linewidth=2, label='Train', marker='o', markersize=3)
        ax.fill_between(epochs, mean_train - std_train, mean_train + std_train, 
                        color='blue', alpha=0.2)
        # Test
        ax.plot(epochs, mean_test, 'r--', linewidth=2, label='Test', marker='s', markersize=3)
        ax.fill_between(epochs, mean_test - std_test, mean_test + std_test, 
                        color='red', alpha=0.2)
        # Add vertical lines at minima
        add_min_vline(ax, epochs, mean_train, 'blue', label='Train min')
        add_min_vline(ax, epochs, mean_test, 'red', label='Test min')
        
        # Add reference line for predicted bias onset if provided (before legend call)
        if epoch_kl_min is not None:
            ax.axvline(epoch_kl_min, color='green', linestyle=':', linewidth=2, alpha=0.8, label='Exp2 bias onset')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xscale('log')
        if use_symlog:
            ax.set_yscale('symlog', linthresh=0.01)
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        else:
            ax.set_yscale('log')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Row 0: Fixed mode
    plot_with_shading(axes[0, 0], epochs, 
                      means['train_loss_p_star_fixed'], stds['train_loss_p_star_fixed'],
                      means['test_loss_p_star_fixed'], stds['test_loss_p_star_fixed'],
                      'p* Loss', f'p* (Distillation) - Fixed t={fixed_t}')
    
    plot_with_shading(axes[0, 1], epochs, 
                      means['train_loss_residual_fixed'], stds['train_loss_residual_fixed'],
                      means['test_loss_residual_fixed'], stds['test_loss_residual_fixed'],
                      'Residual', f'Residual - Fixed t={fixed_t}', use_symlog=True)
    
    # Row 1: Mean mode
    plot_with_shading(axes[1, 0], epochs, 
                      means['train_loss_p_star_mean'], stds['train_loss_p_star_mean'],
                      means['test_loss_p_star_mean'], stds['test_loss_p_star_mean'],
                      'p* Loss', 'p* (Distillation) - Mean over t')
    
    plot_with_shading(axes[1, 1], epochs, 
                      means['train_loss_residual_mean'], stds['train_loss_residual_mean'],
                      means['test_loss_residual_mean'], stds['test_loss_residual_mean'],
                      'Residual', 'Residual - Mean over t', use_symlog=True)
    
    plt.suptitle(f'Multi-Seed Aggregation (n={n_seeds} seeds){title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved multi-seed plot to {save_path}")
    
    return means, stds


def plot_multiseed_overlay(
    all_results: Dict[int, LossDecompositionResults],
    save_path: Path,
):
    """
    Overlay plot for multi-seed: Fixed vs Mean comparison with aggregation.
    Shows p* for both modes on same subplot with shading.
    """
    seeds = sorted(all_results.keys())
    n_seeds = len(seeds)
    epochs = np.array(all_results[seeds[0]].epochs)
    n_epochs = len(epochs)
    fixed_t = all_results[seeds[0]].fixed_timestep
    
    # Aggregate
    p_star_train_fixed = np.zeros((n_seeds, n_epochs))
    p_star_train_mean = np.zeros((n_seeds, n_epochs))
    p_star_test_fixed = np.zeros((n_seeds, n_epochs))
    p_star_test_mean = np.zeros((n_seeds, n_epochs))
    
    for i, seed in enumerate(seeds):
        res = all_results[seed]
        p_star_train_fixed[i, :] = res.train_loss_p_star_fixed
        p_star_train_mean[i, :] = res.train_loss_p_star_mean
        p_star_test_fixed[i, :] = res.test_loss_p_star_fixed
        p_star_test_mean[i, :] = res.test_loss_p_star_mean
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train
    ax = axes[0]
    mean_fixed = p_star_train_fixed.mean(axis=0)
    std_fixed = p_star_train_fixed.std(axis=0)
    mean_mean = p_star_train_mean.mean(axis=0)
    std_mean = p_star_train_mean.std(axis=0)
    
    ax.plot(epochs, mean_fixed, 'b-', linewidth=2.5, label=f'Fixed t={fixed_t}', marker='o', markersize=4)
    ax.fill_between(epochs, mean_fixed - std_fixed, mean_fixed + std_fixed, color='blue', alpha=0.2)
    ax.plot(epochs, mean_mean, 'orange', linestyle='--', linewidth=2.5, label='Mean t', marker='s', markersize=4)
    ax.fill_between(epochs, mean_mean - std_mean, mean_mean + std_mean, color='orange', alpha=0.2)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, mean_fixed, 'blue')
    add_min_vline(ax, epochs, mean_mean, 'orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('p* Loss', fontsize=12)
    ax.set_title('Train: p* Fixed vs Mean', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Test
    ax = axes[1]
    mean_fixed = p_star_test_fixed.mean(axis=0)
    std_fixed = p_star_test_fixed.std(axis=0)
    mean_mean = p_star_test_mean.mean(axis=0)
    std_mean = p_star_test_mean.std(axis=0)
    
    ax.plot(epochs, mean_fixed, 'b-', linewidth=2.5, label=f'Fixed t={fixed_t}', marker='o', markersize=4)
    ax.fill_between(epochs, mean_fixed - std_fixed, mean_fixed + std_fixed, color='blue', alpha=0.2)
    ax.plot(epochs, mean_mean, 'orange', linestyle='--', linewidth=2.5, label='Mean t', marker='s', markersize=4)
    ax.fill_between(epochs, mean_mean - std_mean, mean_mean + std_mean, color='orange', alpha=0.2)
    # Add vertical lines at minima
    add_min_vline(ax, epochs, mean_fixed, 'blue')
    add_min_vline(ax, epochs, mean_mean, 'orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('p* Loss', fontsize=12)
    ax.set_title('Test: p* Fixed vs Mean', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Seed p* Overlay: Fixed t={fixed_t} vs Mean (n={n_seeds} seeds)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved multi-seed overlay plot to {save_path}")


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_loss_decomposition(
    model_dir: str,
    epochs: Optional[List[int]] = None,
    batch_size: int = 128,
    sample_size: Optional[int] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    fixed_timestep_ratio: float = 0.1,
) -> LossDecompositionResults:
    """
    Run loss decomposition analysis on a trained model.
    
    Computes loss decomposition in TWO modes:
    1. Fixed timestep: Single low-noise timestep for 'cleanup' phase analysis
    2. Mean (Expectation): Averaged over all timesteps for global dynamics
    
    Args:
        model_dir: Path to model directory with checkpoints
        epochs: List of epochs to evaluate (None = all available)
        batch_size: Batch size for evaluation
        sample_size: Number of samples to use (None = full dataset)
        seed: Random seed for reproducibility
        device: torch device
        fixed_timestep_ratio: Ratio of t_final for fixed timestep (default 0.1 = 10%)
        
    Returns:
        LossDecompositionResults with all metrics for both modes
    """
    if device is None:
        device = setup_device()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model_dir = Path(model_dir)
    params_path = model_dir / 'full_params.json'
    params = load_params(params_path)
    
    logger.info(f"Loading model from: {model_dir}")
    logger.info(f"Parameters: {json.dumps(params, indent=2)}")
    
    # --- Data Setup ---
    vocab_size, sequences, seq_len, k, rho, _ = load_data(params['data_path'], device)
    
    N_train = params['reduced_length']
    
    # Use pick_i_for_training if present (priority over pick_last_for_training)
    pick_i = params.get('pick_i_for_training', None)
    if pick_i is not None:
        # Block indexing: train = [pick_i*N : (pick_i+1)*N], test = [(pick_i+1)*N : (pick_i+2)*N]
        li = pick_i * N_train
        ri = li + N_train
        total = len(sequences)
        if ri + N_train <= total:
            data_train = sequences[li:ri]
            data_test = sequences[ri:ri + N_train]
            logger.info(f"Data Split: Training block {pick_i} [{li}:{ri}], Test block [{ri}:{ri+N_train}]")
        else:
            # Fallback to pick_i=0 with warning
            logger.warning(f"pick_i={pick_i} out of bounds (total={total}), falling back to pick_i=0")
            data_train = sequences[:N_train]
            data_test = sequences[N_train:2*N_train]
    elif params.get('pick_last_for_training', False):
        data_train = sequences[-N_train:]
        data_test = sequences[:N_train]
        logger.info(f"Data Split: Training on LAST {N_train} samples (pick_last=True)")
    else:
        data_train = sequences[:N_train]
        data_test = sequences[N_train:2*N_train]
        logger.info(f"Data Split: Training on FIRST {N_train} samples")
    
    # Subsample if requested
    if sample_size is not None and sample_size < N_train:
        perm = torch.randperm(N_train)[:sample_size]
        data_train = data_train[perm]
        perm_test = torch.randperm(len(data_test))[:sample_size]
        data_test = data_test[perm_test]
        logger.info(f"Subsampled to {sample_size} samples")
    
    # One-hot encoding
    data_train_oh = nn.functional.one_hot(data_train.long(), vocab_size).float()
    data_test_oh = nn.functional.one_hot(data_test.long(), vocab_size).float()
    
    train_loader = torch.utils.data.DataLoader(data_train_oh, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test_oh, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(data_train)}, Test samples: {len(data_test)}")
    logger.info(f"Vocab size: {vocab_size}, Seq len: {seq_len}, k: {k}, rho: {rho}")
    
    # --- Model & Schedule Setup ---
    model, _, _ = create_model(params, vocab_size, seq_len, device)
    model = torch.compile(model)
    
    t_final = params['t_final']
    alpha_bars = alpha_bars_schedule(
        t_final, device, s=params['s'],
        schedule='linear'
    )
    
    # Calculate fixed timestep (low noise, cleanup phase)
    fixed_timestep = max(1, int(t_final * fixed_timestep_ratio))
    logger.info(f"Fixed timestep: t={fixed_timestep} (ratio={fixed_timestep_ratio}, t_final={t_final})")
    
    # --- Pre-Generate Fixed Noise Maps ---
    # This ensures reproducibility: same noise realization across all epochs
    logger.info("Pre-generating fixed noise maps for reproducibility...")
    
    N_train_samples = len(data_train_oh)
    N_test_samples = len(data_test_oh)
    sample_shape = data_train_oh.shape[1:]  # (L, V)
    
    noise_maps = {
        # Fixed timestep mode - same noise for all epochs
        'train_noise_fixed': torch.randn(N_train_samples, *sample_shape, device=device),
        'test_noise_fixed': torch.randn(N_test_samples, *sample_shape, device=device),
        # Mean timestep mode - same random t and noise for all epochs
        'train_noise_mean': torch.randn(N_train_samples, *sample_shape, device=device),
        'train_t_mean': torch.randint(1, t_final + 1, (N_train_samples,), device=device),
        'test_noise_mean': torch.randn(N_test_samples, *sample_shape, device=device),
        'test_t_mean': torch.randint(1, t_final + 1, (N_test_samples,), device=device),
    }
    logger.info(f"Generated noise maps: train={N_train_samples}, test={N_test_samples}")
    
    # --- Get Epochs to Evaluate ---
    if epochs is None:
        available_epochs = find_available_epochs(model_dir)
    else:
        available_epochs = epochs
    
    available_epochs = sorted(available_epochs)
    logger.info(f"Evaluating {len(available_epochs)} epochs: {available_epochs}")
    
    # --- Evaluation Loop ---
    results = LossDecompositionResults()
    results.fixed_timestep = fixed_timestep
    results.t_final = t_final
    
    for epoch in tqdm(available_epochs, desc="Evaluating epochs"):
        train_metrics, test_metrics = evaluate_epoch(
            model, epoch, train_loader, test_loader,
            alpha_bars, rho, k, vocab_size, t_final, device, model_dir,
            fixed_timestep=fixed_timestep,
            noise_maps=noise_maps,
        )
        
        if train_metrics is None:
            continue
        
        results.epochs.append(epoch)
        
        # Fixed mode
        results.train_loss_total_fixed.append(train_metrics['loss_total_fixed'])
        results.train_loss_p_star_fixed.append(train_metrics['loss_p_star_fixed'])
        results.train_loss_residual_fixed.append(train_metrics['loss_residual_fixed'])
        results.test_loss_total_fixed.append(test_metrics['loss_total_fixed'])
        results.test_loss_p_star_fixed.append(test_metrics['loss_p_star_fixed'])
        results.test_loss_residual_fixed.append(test_metrics['loss_residual_fixed'])
        
        # Mean mode
        results.train_loss_total_mean.append(train_metrics['loss_total_mean'])
        results.train_loss_p_star_mean.append(train_metrics['loss_p_star_mean'])
        results.train_loss_residual_mean.append(train_metrics['loss_residual_mean'])
        results.test_loss_total_mean.append(test_metrics['loss_total_mean'])
        results.test_loss_p_star_mean.append(test_metrics['loss_p_star_mean'])
        results.test_loss_residual_mean.append(test_metrics['loss_residual_mean'])
        
        logger.info(
            f"Epoch {epoch:6d} | "
            f"[Fixed t={fixed_timestep}] Train: total={train_metrics['loss_total_fixed']:.4f}, p*={train_metrics['loss_p_star_fixed']:.4f} | "
            f"Test: total={test_metrics['loss_total_fixed']:.4f}, p*={test_metrics['loss_p_star_fixed']:.4f}"
        )
        logger.info(
            f"           | "
            f"[Mean t] Train: total={train_metrics['loss_total_mean']:.4f}, p*={train_metrics['loss_p_star_mean']:.4f} | "
            f"Test: total={test_metrics['loss_total_mean']:.4f}, p*={test_metrics['loss_p_star_mean']:.4f}"
        )
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def run_multiseed(
    base_dir: str,
    seeds: List[int],
    subpath: str,
    epochs: Optional[List[int]],
    batch_size: int,
    sample_size: Optional[int],
    random_seed: int,
    fixed_timestep_ratio: float,
    device: torch.device,
) -> Dict[int, LossDecompositionResults]:
    """
    Run loss decomposition across multiple model seeds.
    
    Args:
        base_dir: Base directory containing models_restricted_continuous_{seed}/
        seeds: List of seed values to evaluate
        subpath: Subpath after seed directory (e.g., 'test_model_script_30000_12000_cross_entropy_loss/8_512')
        epochs: Epochs to evaluate
        batch_size: Batch size
        sample_size: Sample size
        random_seed: Random seed for reproducibility
        fixed_timestep_ratio: Ratio for fixed timestep
        device: torch device
        
    Returns:
        Dict mapping seed -> LossDecompositionResults
    """
    all_results = {}
    
    for model_seed in seeds:
        model_dir = Path(base_dir) / f"models_restricted_continuous_{model_seed}" / subpath
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found for seed {model_seed}: {model_dir}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing seed {model_seed}")
        logger.info(f"{'='*60}")
        
        try:
            results = run_loss_decomposition(
                model_dir=str(model_dir),
                epochs=epochs,
                batch_size=batch_size,
                sample_size=sample_size,
                seed=random_seed,
                device=device,
                fixed_timestep_ratio=fixed_timestep_ratio,
            )
            all_results[model_seed] = results
        except Exception as e:
            logger.error(f"Error processing seed {model_seed}: {e}")
            continue
    
    return all_results


def main(args: Namespace) -> None:
    """Main entry point."""
    device = setup_device()
    
    # Parse epochs if provided
    epochs = None
    if args.epochs:
        epochs = [int(e.strip()) for e in args.epochs.split(',')]
    
    # Parse seeds if provided
    model_seeds = None
    if args.seeds:
        model_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # ========== SELECTION MODE (multi-seed) ==========
    selected_subpath = None
    predicted_epoch_kl_min = None
    selection_gap_log = None
    ranking = []  # For diagnostic plot
    seq_len_for_kl = None  # Will be set if selection mode is used
    
    if args.select_by_exp2 and args.base_dir and model_seeds:
        if not HAS_NN_SELECTOR:
            logger.error("--select-by-exp2 requires nn_divergence_selector module")
            return
        
        # Parse candidate subpaths
        if args.candidate_subpaths:
            candidate_subpaths = [s.strip() for s in args.candidate_subpaths.split(',')]
        else:
            candidate_subpaths = [args.subpath]  # Default to single subpath
        
        # Need seq_len for KL computation - load from first available model
        first_model_dir = Path(args.base_dir) / f"models_restricted_continuous_{model_seeds[0]}" / candidate_subpaths[0]
        if first_model_dir.exists():
            first_params = load_params(first_model_dir / "full_params.json")
            _, seqs_temp, seq_len_for_kl, _, _, _ = load_data(first_params['data_path'], 'cpu')
        else:
            logger.error(f"Cannot find first model to get seq_len: {first_model_dir}")
            return
        
        logger.info(f"\nRunning Exp2 gap selection on {len(candidate_subpaths)} candidates...")
        best_subpath, mean_gap, mean_kl_min, ranking = rank_candidates(
            base_dir=Path(args.base_dir),
            seeds=model_seeds,
            candidate_subpaths=candidate_subpaths,
            n_test=args.exp2_n_test,
            nbins=args.exp2_nbins,
            seq_len=seq_len_for_kl,
        )
        
        if best_subpath is None:
            logger.error("No valid candidates found! Exiting.")
            return
        
        selected_subpath = best_subpath
        predicted_epoch_kl_min = mean_kl_min
        selection_gap_log = mean_gap
        
        logger.info(f"\nSELECTED: {selected_subpath} (gap_log={selection_gap_log:.4f}, predicted KL min @ epoch {predicted_epoch_kl_min:.0f})")
        
        if args.select_only:
            logger.info("--select-only mode: exiting without Exp9 evaluation")
            return
        
        # Override subpath with selected one
        args.subpath = selected_subpath
    
    # ========== MULTI-SEED MODE ==========
    if args.base_dir and model_seeds:
        # Confirm final subpath being used (important when selection mode is active)
        final_subpath = args.subpath
        logger.info(f"Running in MULTI-SEED mode with seeds: {model_seeds}")
        logger.info(f"USING SUBPATH: {final_subpath}")
        if selected_subpath is not None:
            assert args.subpath == selected_subpath, f"Subpath mismatch: {args.subpath} != {selected_subpath}"
            logger.info(f"  (auto-selected by Exp2 gap with gap_log={selection_gap_log:.4f})")
        all_results = run_multiseed(
            base_dir=args.base_dir,
            seeds=model_seeds,
            subpath=args.subpath,
            epochs=epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            random_seed=args.seed,
            fixed_timestep_ratio=args.fixed_t_ratio,
            device=device,
        )
        
        if len(all_results) == 0:
            logger.error("No seeds were successfully evaluated!")
            return
        
        logger.info(f"\nSuccessfully evaluated {len(all_results)} seeds: {list(all_results.keys())}")
        
        # --- Save aggregated results ---
        # Always save to repo-local plots directory, not base_dir
        fixed_t = all_results[list(all_results.keys())[0]].fixed_timestep
        run_config = f"N{args.sample_size or 'full'}_t{fixed_t}_seed{args.seed}"
        _out_base = Path(args.output_root) if args.output_root else PROJECT_ROOT / 'plots'
        output_dir = _out_base / 'exp9' / run_config
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        seeds_str = "_".join(map(str, sorted(all_results.keys())))
        
        # Save individual seed results as JSON
        for seed, res in all_results.items():
            json_path = output_dir / f'seed{seed}_N{args.sample_size or "full"}.json'
            with open(json_path, 'w') as f:
                json.dump(res.to_dict(), f, indent=2)
        
        # Aggregate and save
        agg_epochs = all_results[list(all_results.keys())[0]].epochs
        agg_data = {
            'seeds': list(all_results.keys()),
            'epochs': agg_epochs,
            'fixed_timestep': all_results[list(all_results.keys())[0]].fixed_timestep,
        }
        
        # Add selection metadata if available
        if selected_subpath is not None:
            agg_data['selected_subpath'] = selected_subpath
            agg_data['selection_gap_log'] = selection_gap_log
            agg_data['predicted_epoch_kl_min'] = predicted_epoch_kl_min
        
        for key in ['train_loss_total_fixed', 'train_loss_p_star_fixed', 'train_loss_residual_fixed',
                    'test_loss_total_fixed', 'test_loss_p_star_fixed', 'test_loss_residual_fixed',
                    'train_loss_total_mean', 'train_loss_p_star_mean', 'train_loss_residual_mean',
                    'test_loss_total_mean', 'test_loss_p_star_mean', 'test_loss_residual_mean']:
            vals = np.array([getattr(all_results[s], key) for s in sorted(all_results.keys())])
            agg_data[f'{key}_mean'] = vals.mean(axis=0).tolist()
            agg_data[f'{key}_std'] = vals.std(axis=0).tolist()
        
        agg_json_path = output_dir / f'aggregated_seeds{seeds_str}_N{args.sample_size or "full"}.json'
        with open(agg_json_path, 'w') as f:
            json.dump(agg_data, f, indent=2)
        logger.info(f"Saved aggregated results to {agg_json_path}")
        
        # --- Generate Plots ---
        if args.plots:
            # Multi-seed aggregation plot
            plot_multiseed_aggregation(
                all_results,
                output_dir / f'multiseed_aggregation_seeds{seeds_str}_N{args.sample_size or "full"}.png',
                title_suffix=f' (Seeds: {list(all_results.keys())})',
                epoch_kl_min=predicted_epoch_kl_min,
            )
            
            # Multi-seed overlay plot
            plot_multiseed_overlay(
                all_results,
                output_dir / f'multiseed_overlay_seeds{seeds_str}_N{args.sample_size or "full"}.png',
            )
            
            # Exp2 diagnostic plot (if selection mode was used)
            if selected_subpath is not None and HAS_NN_SELECTOR:
                exp2_data = collect_exp2_data_for_plot(
                    base_dir=Path(args.base_dir),
                    seeds=model_seeds,
                    subpath=selected_subpath,
                    n_test=args.exp2_n_test,
                    nbins=args.exp2_nbins,
                    seq_len=seq_len_for_kl,
                )
                if exp2_data is not None:
                    # Get mean epoch_test_min from ranking
                    mean_test_min = ranking[0].get('mean_epoch_test_min', None) if ranking else None
                    plot_exp2_diagnostic(
                        exp2_data,
                        epoch_kl_min=predicted_epoch_kl_min,
                        epoch_test_min=mean_test_min or predicted_epoch_kl_min * 5,  # fallback
                        save_path=output_dir / f'exp2_diagnostic_seeds{seeds_str}.png',
                        title_suffix=f' ({selected_subpath})',
                    )
            
            # Also plot individual seeds
            for seed, res in all_results.items():
                plot_fixed_vs_mean_overlay(
                    res,
                    output_dir / f'fixed_vs_mean_overlay_seed{seed}_N{args.sample_size or "full"}.png',
                )
        
        logger.info("Multi-seed experiment completed successfully!")
        return
    
    # ========== SINGLE-SEED MODE ==========
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return
    
    # Run evaluation
    results = run_loss_decomposition(
        model_dir=str(model_dir),
        epochs=epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        seed=args.seed,
        device=device,
        fixed_timestep_ratio=args.fixed_t_ratio,
    )
    
    if len(results.epochs) == 0:
        logger.error("No epochs were successfully evaluated!")
        return
    
    # --- Save Results ---
    # Include run configuration in path to avoid rsync overwrites
    run_config = f"N{args.sample_size or 'full'}_t{results.fixed_timestep}_seed{args.seed}"
    _out_base = Path(args.output_root) if args.output_root else PROJECT_ROOT / 'plots'
    output_dir = _out_base / 'exp9' / f'exp9_{run_config}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / f'loss_decomposition_seed{args.seed}_N{args.sample_size or "full"}.json'
    with open(json_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    logger.info(f"Saved results to {json_path}")
    
    # Save NPZ
    npz_path = output_dir / f'loss_decomposition_seed{args.seed}_N{args.sample_size or "full"}.npz'
    np.savez(
        npz_path,
        epochs=np.array(results.epochs),
        fixed_timestep=results.fixed_timestep,
        t_final=results.t_final,
        # Fixed mode
        train_loss_total_fixed=np.array(results.train_loss_total_fixed),
        train_loss_p_star_fixed=np.array(results.train_loss_p_star_fixed),
        train_loss_residual_fixed=np.array(results.train_loss_residual_fixed),
        test_loss_total_fixed=np.array(results.test_loss_total_fixed),
        test_loss_p_star_fixed=np.array(results.test_loss_p_star_fixed),
        test_loss_residual_fixed=np.array(results.test_loss_residual_fixed),
        # Mean mode
        train_loss_total_mean=np.array(results.train_loss_total_mean),
        train_loss_p_star_mean=np.array(results.train_loss_p_star_mean),
        train_loss_residual_mean=np.array(results.train_loss_residual_mean),
        test_loss_total_mean=np.array(results.test_loss_total_mean),
        test_loss_p_star_mean=np.array(results.test_loss_p_star_mean),
        test_loss_residual_mean=np.array(results.test_loss_residual_mean),
    )
    logger.info(f"Saved NPZ to {npz_path}")
    
    # --- Print Summary Table for Quick Verification ---
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY (for quick verification)")
    logger.info("="*80)
    logger.info(f"{'Epoch':>8} | {'Train Total (Fixed)':>18} | {'Train p* (Fixed)':>16} | {'Test Total (Fixed)':>17} | {'Test p* (Fixed)':>15}")
    logger.info("-"*80)
    for i, ep in enumerate(results.epochs):
        logger.info(f"{ep:>8} | {results.train_loss_total_fixed[i]:>18.4f} | {results.train_loss_p_star_fixed[i]:>16.4f} | {results.test_loss_total_fixed[i]:>17.4f} | {results.test_loss_p_star_fixed[i]:>15.4f}")
    logger.info("="*80 + "\n")
    
    # --- Generate Plots ---
    if args.plots:
        plot_loss_decomposition(
            results,
            output_dir / f'loss_decomposition_seed{args.seed}_N{args.sample_size or "full"}.png',
            title_suffix=f' (Seed={args.seed}, N={args.sample_size or "full"})'
        )
        
        plot_combined_view(
            results,
            output_dir / f'loss_decomposition_combined_seed{args.seed}_N{args.sample_size or "full"}.png',
        )
        
        # NEW: Fixed vs Mean overlay
        plot_fixed_vs_mean_overlay(
            results,
            output_dir / f'fixed_vs_mean_overlay_seed{args.seed}_N{args.sample_size or "full"}.png',
        )
    
    logger.info("Experiment 9 completed successfully!")


if __name__ == '__main__':
    parser = ArgumentParser(description="Experiment 9: Loss Decomposition Analysis")
    
    # Single-seed mode
    parser.add_argument(
        '--model-dir', type=str, default=None,
        help='Path to model directory with checkpoints (single-seed mode)'
    )
    
    # Multi-seed mode
    parser.add_argument(
        '--base-dir', type=str, default=None,
        help='Base directory containing models_restricted_continuous_{seed}/ (multi-seed mode)'
    )
    parser.add_argument(
        '--seeds', type=str, default=None,
        help='Comma-separated list of model seeds to evaluate (e.g., "2,3,4,5,6")'
    )
    parser.add_argument(
        '--subpath', type=str, 
        default='test_model_script_30000_12000_cross_entropy_loss/8_512',
        help='Subpath after seed directory'
    )
    
    # Selection mode arguments
    parser.add_argument(
        '--select-by-exp2', action='store_true',
        help='Auto-select best config using Exp2 KL gap metric'
    )
    parser.add_argument(
        '--candidate-subpaths', type=str, default=None,
        help='Comma-separated subpaths to consider for selection (multi-seed mode)'
    )
    parser.add_argument(
        '--candidate-model-dirs', type=str, default=None,
        help='Comma-separated model dirs to consider for selection (single-seed mode)'
    )
    parser.add_argument(
        '--exp2-n-test', type=int, default=100,
        help='N_test used by Exp2 run (must match cache)'
    )
    parser.add_argument(
        '--exp2-nbins', type=int, default=20,
        help='nbins used by Exp2 histogram (must match cache)'
    )
    parser.add_argument(
        '--select-only', action='store_true',
        help='Run selection and print ranking, then exit without Exp9 evaluation'
    )
    
    # Common arguments
    parser.add_argument(
        '--epochs', type=str, default=None,
        help='Comma-separated list of epochs to evaluate. Omit to scan all '
             'available checkpoints (recommended: the full trajectory is '
             'needed to locate the NN-divergence and test-loss minima).'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Batch size for evaluation (default: 128)'
    )
    parser.add_argument(
        '--sample-size', type=int, default=None,
        help='Number of samples to use (default: full dataset)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--fixed-t-ratio', type=float, default=0.3,
        help='Ratio of t_final for fixed timestep mode (default: 0.3 = t=150 if t_final=500)'
    )
    parser.add_argument(
        '--plots', action='store_true',
        help='Generate and save plots'
    )
    parser.add_argument(
        '--output-root', type=str, default=None,
        help='Root dir for output plots (default: model_dir/plots or PROJECT_ROOT/plots_exp9)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_dir and not (args.base_dir and args.seeds):
        parser.error("Either --model-dir (single-seed) or --base-dir with --seeds (multi-seed) is required")
    
    main(args)
