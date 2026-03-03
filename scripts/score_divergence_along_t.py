"""
score_divergence_along_t.py - Factorized BP trajectory analysis
Referred to as exp5_new in output directory and plotting functions.
Reproduces Fig. 3(b) (KL between BP_k and trained model vs diffusion time).
==========================================================================

Measures the KL divergence between the trained model and hierarchically
filtered BP approximations (BP_k, k=0,...,4) as a function of the denoising
timestep t. By comparing at different factorization levels k, we can identify
which depth of the generative tree the model is effectively approximating at
each stage of the reverse diffusion process.

In the paper, the peak around t/T ~ 0.15 defines the "critical time" used
for fixed-time score comparisons in sequential_learning.py.

Procedure:
1. Identify pairs of model checkpoints.
2. For a fixed epoch, noise test samples to t_max.
3. For each timestep t from t_max down to 1:
   - Compute model prediction.
   - For each factorization level k, compute BP_k prediction.
   - Compute KL(BP_k vs Model).
4. Average over model pairs and noise repetitions.

Outputs:
- Plots showing KL trajectories for different k values.
- NPZ files containing raw KL data.

Usage (run from scripts/ directory)::

    # cross-size comparison (Fig. 3(b), n=5k/12k/70k)
    python score_divergence_along_t.py \\
        --output-root ../plots/ \\
        --base-path "<base>/results_transformer_pick_{}_for_training" \\
        --train-size 5000 --train-size-2 12000 --train-size-3 70000 \\
        --N-test 2000 \\
        --bp-factorized-layers "0,1,2" \\
        --data-ids "0,0,1" --same-dataset 0 \\
        --which-epoch1 <E1> --which-epoch2 <E2> --which-epoch3 <E3>

    Epoch selection: Fig. 3(b) evaluates each model at its test-loss
    minimum. The --which-epoch{1,2,3} flags are required; run
    nn_divergence.py or loss_decomposition.py first to identify the
    test-loss minimum for each training size.
"""
#!/usr/bin/env python3
import logging
from argparse import ArgumentParser
from pathlib import Path
import itertools
import math
import tempfile
import os
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.diffusion import forward_process, backward_process, backward_process_gt, backward_process_gt_torch
from utils import (
    load_params,
    load_data,
    setup_device,
    create_model,
    load_checkpoint,
    kl_divergence_batch,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "grid.alpha": 0.3,
    }
)


##### Helper functions #####

def _atomic_savefig(fig, out_path: Path, **savefig_kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=out_path.suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        fig.savefig(tmp_path, **savefig_kwargs)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

def _atomic_savez(out_path: Path, **npz_kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=out_path.suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(tmp_path, **npz_kwargs)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

def dataset_key(exp: dict) -> str:
    return str(exp["params"]["data_path"])

def discover_pairs(base_path_fmt: str, same_dataset: int, train_size: int, num_datasets: int = 13):
    """
    For each data_id in [0..num_datasets-1]:
      - root = base_path_fmt.format(data_id)
      - find leaf dirs containing f"_{train_size}" in their path
      - pick 2 leaf dirs (two seeds) per dataset
    Then build:
      - same_dataset==1 : pair the two seeds within each dataset
      - else: across datasets, all dataset pairs, all 2x2 seed combinations
    """
    all_dirs = []

    # Find all paths of the model with a certain training size
    for data_id in range(int(num_datasets)):
        root = Path(base_path_fmt.format(data_id))
        if not root.exists():
            logger.info(f"Root path not found: {root}")
            continue
        
        dirs_with_train_size = [
            d for d in root.rglob("*")
            if d.is_dir() and ("_" + str(train_size)) in str(d)
        ]
        leaf_dirs = [d for d in dirs_with_train_size if not any(x.is_dir() for x in d.iterdir())]

        if len(leaf_dirs) > 2:
            all_dirs.append(leaf_dirs[-2:])
        elif len(leaf_dirs) == 2:
            all_dirs.append(leaf_dirs)
        elif len(leaf_dirs) == 1:
            # Special case: only 1 seed found, pair it with itself
            logger.warning(f"Only 1 leaf dir found in {root}. Pairing with itself.")
            all_dirs.append([leaf_dirs[0], leaf_dirs[0]])
        else:
            logger.info(f"Found {len(leaf_dirs)} leaf dirs (need >= 1) in {root} with pattern *{train_size}*")
            continue

    if len(all_dirs) == 0:
        raise RuntimeError("No eligible experiment directories found.")

    pairs_dir = np.array(all_dirs, dtype=object)

    # Build pairs
    if int(same_dataset) == 1:
        # If same dataset, pair together only models trained on the same dataset
        pairs = [(pairs_dir[i, 0], pairs_dir[i, 1]) for i in range(pairs_dir.shape[0])]
    else:
        # If not same dataset, pair together only models trained on the same dataset
        pairs = [
            (pairs_dir[i, s1], pairs_dir[j, s2])
            for (i, j) in itertools.combinations(range(len(all_dirs)), 2)
            for s1 in range(2)
            for s2 in range(2)
        ]

    return all_dirs, pairs


def discover_cross_size_pairs(
    base_path_fmt: str, 
    train_size_1: int, 
    train_size_2: int, 
    num_datasets: int = 1,
    max_seeds: int = 5,
) -> list:
    """
    Discover pairs of models where Model1 is trained on train_size_1 and 
    Model2 is trained on train_size_2, from the same data pick.
    This is needed to compare 5k vs 12k trained models on the same plot.
    
    Args:
        max_seeds: Maximum number of seed pairs to return (for efficiency).
    
    Returns:
        List of tuples: [(path_5k, path_12k), ...]
    """
    pairs = []
    
    for data_id in range(int(num_datasets)):
        root = Path(base_path_fmt.format(data_id))
        if not root.exists():
            logger.info(f"Root path not found: {root}")
            continue
        
        # Find all seed directories
        seed_dirs = sorted([d for d in root.iterdir() if d.is_dir() and "models_restricted_continuous_" in str(d)])
        
        for seed_dir in seed_dirs:
            if len(pairs) >= max_seeds:
                break
                
            # Find model with train_size_1
            dirs_size1 = [
                d for d in seed_dir.rglob("*")
                if d.is_dir() and (f"_{train_size_1}_" in str(d) or str(d).endswith(f"_{train_size_1}"))
            ]
            leaf_dirs_1 = [d for d in dirs_size1 if d.exists() and not any(x.is_dir() for x in d.iterdir() if x.exists())]
            
            # Find model with train_size_2
            dirs_size2 = [
                d for d in seed_dir.rglob("*")
                if d.is_dir() and (f"_{train_size_2}_" in str(d) or str(d).endswith(f"_{train_size_2}"))
            ]
            leaf_dirs_2 = [d for d in dirs_size2 if d.exists() and not any(x.is_dir() for x in d.iterdir() if x.exists())]
            
            if leaf_dirs_1 and leaf_dirs_2:
                # Pair first match of each
                pairs.append((leaf_dirs_1[0], leaf_dirs_2[0]))
                logger.info(f"Found cross-size pair: {leaf_dirs_1[0].name} vs {leaf_dirs_2[0].name}")
        
        if len(pairs) >= max_seeds:
            break
    
    if not pairs:
        raise RuntimeError(f"No cross-size pairs found for sizes {train_size_1} and {train_size_2}")
    
    logger.info(f"Using {len(pairs)} seed pairs (max_seeds={max_seeds})")
    return pairs


def discover_multi_size_models(
    base_path_fmt: str,
    train_sizes: list,  # e.g. [5000, 12000, 70000]
    data_ids: list = None,  # e.g. [0, 0, 1] - which pick for each size
    max_seeds: int = 5,
) -> dict:
    """
    Discover models for multiple train sizes, potentially from different data picks.
    
    Args:
        base_path_fmt: Path with {} placeholder for data_id
        train_sizes: List of train sizes to find (e.g. [5000, 12000, 70000])
        data_ids: Which data_id (pick) to use for each train size. 
                  If None, uses pick 0 for all.
        max_seeds: Maximum seeds to find per size
    
    Returns:
        Dict mapping train_size -> list of model paths
    """
    if data_ids is None:
        data_ids = [0] * len(train_sizes)
    
    result = {}
    
    for train_size, data_id in zip(train_sizes, data_ids):
        root = Path(base_path_fmt.format(data_id))
        if not root.exists():
            logger.warning(f"Root path not found: {root}")
            result[train_size] = []
            continue
        
        # Find all seed directories
        seed_dirs = sorted([d for d in root.iterdir() if d.is_dir() and "models_restricted_continuous_" in str(d)])
        
        models_found = []
        for seed_dir in seed_dirs:
            if len(models_found) >= max_seeds:
                break
            
            # Find model with this train_size
            dirs_with_size = [
                d for d in seed_dir.rglob("*")
                if d.is_dir() and (f"_{train_size}_" in str(d) or str(d).endswith(f"_{train_size}"))
            ]
            leaf_dirs = [d for d in dirs_with_size if d.exists() and not any(x.is_dir() for x in d.iterdir() if x.exists())]
            
            if leaf_dirs:
                models_found.append(leaf_dirs[0])
                logger.info(f"Found {train_size}: {leaf_dirs[0]}")
        
        result[train_size] = models_found
        logger.info(f"Found {len(models_found)} models for train_size={train_size} (data_id={data_id})")
    
    return result

def get_schedule(schedule_cache: dict, *, t_test: int, s: float, device):
    key = (int(t_test), float(s), device.type)
    if key in schedule_cache:
        # Retrieve if scheduler is in cache
        return schedule_cache[key]
    
    # Create schedulers and store them in cache (including numpy versions)
    alpha_bars = alpha_bars_schedule(
        int(t_test),
        device=device,
        s=float(s),
        schedule="linear",
    )
    alphas, _ = get_alpha_beta_from_alpha_bar(alpha_bars)

    out = {
        "alpha_bars": alpha_bars,
        "alphas": alphas,
        "alpha_bars_np": alpha_bars.detach().cpu().numpy(),
        "alphas_np": alphas.detach().cpu().numpy(),
    }
    schedule_cache[key] = out
    return out

def build_or_get_experiment(exp_cache: dict, path: Path, device, do_compile: bool):
    # Get model path
    path = Path(path)
    if path in exp_cache:
        # Retrieve experiment if in cache
        return exp_cache[path]

    # If not in cache, load model params
    params = load_params(path / "full_params.json")

    # Load data and create model
    vocab_size, sequences, seq_len, k, rho, _ = load_data(params["data_path"], device)
    model, _, _ = create_model(params, vocab_size, seq_len, device)

    if do_compile:
        model = torch.compile(model)
    model.eval()

    exp_cache[path] = dict(
        path=path,
        params=params,
        vocab_size=vocab_size,
        sequences=sequences,
        seq_len=seq_len,
        k=k,
        rho=rho,
        model=model,
    )
    return exp_cache[path]

def load_epoch_weights_once(model_path: Path, exp_cache: dict, epoch: int, device):
    exp = exp_cache[model_path]
    ckpt = model_path / f"test_model_script_epoch_{epoch}.pt"

    if (not ckpt.exists()) or (not load_checkpoint(exp["model"], str(ckpt), device, training=False)):
        raise RuntimeError(f"Checkpoint load failed: {ckpt}")
    
    exp["model"].eval()

def train_idx_from_pick_i(
    *,
    num_seqs: int,
    reduced_length: int,
    pick_i: int,
    device,) -> Tuple[torch.Tensor, int]:
    """
    Training block chosen by pick_i:
      idx = [pick_i*reduced_length : (pick_i+1)*reduced_length)
    If pick_i is invalid/out-of-bounds, fallback to 0 deterministically.
    """
    reduced_length = int(reduced_length)
    if reduced_length <= 0:
        raise ValueError(f"reduced_length must be >0, got {reduced_length}")
    max_blocks = num_seqs // reduced_length
    if max_blocks <= 0:
        raise ValueError(f"Not enough sequences ({num_seqs}) for reduced_length={reduced_length}")

    pick_i = int(pick_i)
    if pick_i < 0 or pick_i >= max_blocks:
        logger.warning("pick_i_for_training=%d out of bounds (max_blocks=%d). Falling back to 0.", pick_i, max_blocks)
        pick_i = 0

    left = pick_i * reduced_length
    right = left + reduced_length
    if right > num_seqs:
        # Should not happen if max_blocks computed as floor-div
        logger.warning("Computed train block out of bounds. Falling back to 0.")
        left, right, pick_i = 0, reduced_length, 0

    return torch.arange(left, right, device=device), pick_i

def build_shared_test_idx_disjoint_by_content(
    *,
    sequences: torch.Tensor,
    train_idx1: torch.Tensor,
    train_idx2: torch.Tensor,
    N_test: int,
    seed: int,
) -> torch.Tensor:
    """
    Shared test set:
      - disjoint from both train blocks by index
      - and disjoint by content (exact sequence match) from the union of training sequences
    """
    num_seqs = sequences.shape[0]
    all_idx = torch.arange(num_seqs, device=sequences.device)

    train_union = torch.unique(torch.cat([train_idx1, train_idx2], dim=0))
    mask_not_train = torch.ones(num_seqs, dtype=torch.bool, device=sequences.device)
    mask_not_train[train_union] = False
    candidate_idx = all_idx[mask_not_train]
    if candidate_idx.numel() == 0:
        raise RuntimeError("No candidate indices left outside both training blocks.")

    sequences_cpu = sequences.detach().cpu().numpy()
    train_idx1_np = train_idx1.detach().cpu().numpy()
    train_idx2_np = train_idx2.detach().cpu().numpy()
    cand_idx_np = candidate_idx.detach().cpu().numpy()

    train_all_np = np.concatenate([sequences_cpu[train_idx1_np], sequences_cpu[train_idx2_np]], axis=0)
    train_set = {tuple(row) for row in train_all_np}

    valid_mask = np.array([tuple(sequences_cpu[i]) not in train_set for i in cand_idx_np], dtype=bool)
    valid_candidate_idx_np = cand_idx_np[valid_mask]

    if valid_candidate_idx_np.size < N_test:
        raise RuntimeError(
            f"Not enough content-disjoint sequences: needed {N_test}, found {valid_candidate_idx_np.size}."
        )

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(valid_candidate_idx_np)

    return torch.from_numpy(valid_candidate_idx_np[:N_test]).to(sequences.device)


##### Caches #####

def get_noised_stack(
    noised_cache: dict,
    *,
    x0_one_hot: torch.Tensor,
    t_max: int,
    sched: dict,
    seed: int,
) -> torch.Tensor:
    # Noised data key: pointer to clean data, shape, t_max noising time, seed and pointer to alpha_bar
    key = (
        int(x0_one_hot.data_ptr()),
        tuple(x0_one_hot.shape),
        int(t_max),
        int(seed),
        int(sched["alpha_bars"].data_ptr()),
    )
    if key in noised_cache:
        # Retrieve if in cache
        return noised_cache[key]
    
    # Set seeds
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    # Noise clean data and store in cache
    xs = [x0_one_hot]
    for t in range(1, int(t_max) + 1):
        t_tensor = torch.tensor(t, device=x0_one_hot.device, dtype=torch.long)
        xs.append(forward_process(x0_one_hot, t_tensor, alpha_bars=sched["alpha_bars"]))
    out = torch.stack(xs, dim=0)
    noised_cache[key] = out
    return out

def get_bp_preds_trajectory(
    bp_cache: dict,
    *,
    x_noised_stack: torch.Tensor,
    t_max: int,
    rho: Any,
    k: Any,
    vocab_size: int,
    sched: dict,
    device,
    bp_backend: str = 'torch',
    factorized_layers: int = 0,
) -> np.ndarray:

    # BP cache key, indexed by pointer to x_noised data, t_max, bp params, alpha_bars pointer, and factorized_layers
    key = (
        int(x_noised_stack.data_ptr()),
        int(t_max),
        int(k),
        int(vocab_size),
        int(sched["alpha_bars"].data_ptr()),
        int(factorized_layers),
    )
    if key in bp_cache:
        # Retrieve if in cache
        return bp_cache[key]

    batch, seq_len, v = x_noised_stack.shape[1], x_noised_stack.shape[2], x_noised_stack.shape[3]
    assert v == vocab_size

    # Get BP preds for all diffusion times backwards, cache later
    bp_preds = np.empty((t_max, batch, seq_len, vocab_size), dtype=np.float32)
    
    # Select backend function
    if bp_backend == 'torch':
        bp_func = backward_process_gt_torch
    elif bp_backend == 'numpy':
        bp_func = backward_process_gt
    else:
        raise ValueError(f"Unknown bp_backend: {bp_backend}. Use 'torch' or 'numpy'.")
    
    for t in range(int(t_max), 0, -1):
        idx = t - 1
        x_t = x_noised_stack[t].to(device)
        _, x0_hat = bp_func(
            x_t,
            int(t),
            (rho, k, vocab_size),
            device,
            alpha_bars=sched["alpha_bars"],
            alphas=sched["alphas"],
            fix_noise=False,
            factorized_layers=int(factorized_layers),
        )
        bp_preds[idx] = x0_hat.detach().cpu().numpy()

    bp_cache[key] = bp_preds
    return bp_preds

def denoise_trajectory_cached(
    pred_cache: dict,
    *,
    model_path: Path,
    model: torch.nn.Module,
    x_noised_stack: torch.Tensor,
    t_max: int,
    t_final: int,
    sched: dict,
    device,
    split: str,
    epoch: int,
) -> np.ndarray:
    # Key is model path, epoch checkpoint, split (train or test), pointer to noised data, t_max, t_final (scheduler), alpha_bars_pointer
    key = (
        str(model_path),
        int(epoch),
        split,
        int(x_noised_stack.data_ptr()),
        int(t_max),
        int(t_final),
        int(sched["alpha_bars"].data_ptr()),
    )
    if key in pred_cache:
        # Retrieve if in cache
        return pred_cache[key]

    batch, seq_len, vocab_size = x_noised_stack.shape[1], x_noised_stack.shape[2], x_noised_stack.shape[3]
    preds = np.empty((t_max, batch, seq_len, vocab_size), dtype=np.float32)

    # Run inference backwards one time step at a time, store in cache later
    with torch.inference_mode():
        for t in range(int(t_max), 0, -1):
            idx = t - 1
            x_t = x_noised_stack[t]
            _, x0_hat = backward_process(
                x_t,
                int(t),
                int(t_final),
                model,
                device,
                alpha_bars=sched["alpha_bars"],
                alphas=sched["alphas"],
                clamp=False,
                temperature=None,
                fix_noise=False,
            )
            preds[idx] = x0_hat.detach().cpu().numpy()

    pred_cache[key] = preds
    return preds

##### Plotting #####

def plot_factorized_trajectories(
    *,
    plots_dir: Path,
    timesteps: np.ndarray,
    f_list: List[int],
    kl_results: Dict[int, np.ndarray],  # map k -> kl_array of shape (t_max, batch)
    n_pairs_done: int,
    n_pairs_total: int,
    epoch: int,
    tag: str = "latest",
):
    """
    Plots KL divergence trajectories for different factorization levels.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use a colormap
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(f_list), vmax=max(f_list) if max(f_list) > 0 else 1)
    
    for k in sorted(f_list):
        kl_data = kl_results[k]
        mean_kl = np.mean(kl_data, axis=1)
        se_kl = np.std(kl_data, axis=1, ddof=1) / np.sqrt(kl_data.shape[1])
        
        # Color based on k
        color = cmap(norm(k))
        label = f"BP (k={k}) vs Model"
        if k == 0:
            label += " (Exact)"
            color = "black"  # Highlight exact BP
            
        ax.plot(timesteps, mean_kl, lw=2, color=color, label=label)
        ax.fill_between(timesteps, mean_kl - se_kl, mean_kl + se_kl, alpha=0.2, color=color)

    ax.set_title(f"KL(BP_k || Model 1) Trajectories\nPairs: {n_pairs_done}/{n_pairs_total} | Epoch: {epoch}")
    ax.set_xlabel("Denoising step index (t-1)")
    ax.set_ylabel("KL divergence")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend()
    
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    base = f"KL_FACTORIZED_epoch_{epoch}_pairs_{n_pairs_total}_{tag}"
    out_png = plots_dir / f"{base}.png"
    out_npz = plots_dir / f"{base}.npz"

    _atomic_savefig(fig, out_png, bbox_inches="tight")
    plt.close(fig)
    
    # Save raw data
    save_dict = {"timesteps": timesteps}
    for k, data in kl_results.items():
        save_dict[f"kl_bp_k{k}_vs_model"] = data
        
    _atomic_savez(out_npz, **save_dict)
    logger.info("Saved factorized plot: %s", out_png)
    logger.info("Saved factorized data: %s", out_npz)


def plot_gap_analysis(
    *,
    plots_dir: Path,
    timesteps: np.ndarray,
    bp_gaps: Dict[int, np.ndarray],  # map k -> KL(BP_0 || BP_k) of shape (t_max, batch)
    model_gaps: Dict[str, np.ndarray],  # map model_name -> KL(BP_0 || Model) of shape (t_max, batch)
    n_pairs_done: int,
    n_pairs_total: int,
    epochs_used: Dict[str, int],  # map model_name -> epoch evaluated
    train_size: int,
    train_size_2: int = None,  # For cross-size mode title
    model_labels: Dict[str, str] = None,  # Custom labels for models
    tag: str = "latest",
):
    """
    Plots gap analysis: 6 curves total.
    - 4 curves: KL(BP_0 || BP_k) for k in {1,2,3,4} (approximation gap)
    - 2 curves: KL(BP_0 || Model1/Model2) (learning gap)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use Plasma colormap for BP gaps (sequential)
    cmap = plt.get_cmap("plasma")
    k_values = sorted(bp_gaps.keys())
    if k_values:
        norm = plt.Normalize(vmin=min(k_values), vmax=max(k_values))
    
    # Plot BP Approximation Gaps (k=1,2,3,4)
    for k in k_values:
        kl_data = bp_gaps[k]
        mean_kl = np.mean(kl_data, axis=1)
        se_kl = np.std(kl_data, axis=1, ddof=1) / np.sqrt(kl_data.shape[1]) if kl_data.shape[1] > 1 else np.zeros_like(mean_kl)
        
        color = cmap(norm(k))
        label = f"KL(BP₀ || BP_{k})"
            
        ax.plot(timesteps, mean_kl, lw=2, color=color, label=label)
        ax.fill_between(timesteps, mean_kl - se_kl, mean_kl + se_kl, alpha=0.15, color=color)

    # Plot Model Gaps with dotted lines for visual distinction
    model_colors = {"Model1": "crimson", "Model2": "black", "Model3": "forestgreen"}
    model_linestyles = {"Model1": ":", "Model2": ":", "Model3": ":"}  # All dotted for distinction
    
    # Use custom labels if provided
    if model_labels is None:
        model_labels = {"Model1": "Model1", "Model2": "Model2", "Model3": "Model3"}
    
    for model_name, kl_data in model_gaps.items():
        mean_kl = np.mean(kl_data, axis=1)
        se_kl = np.std(kl_data, axis=1, ddof=1) / np.sqrt(kl_data.shape[1]) if kl_data.shape[1] > 1 else np.zeros_like(mean_kl)
        
        color = model_colors.get(model_name, "gray")
        linestyle = model_linestyles.get(model_name, ":")
        epoch = epochs_used.get(model_name, "?")
        display_label = model_labels.get(model_name, model_name)
        label = f"KL(BP₀ || {display_label}) [ep={epoch}]"
            
        ax.plot(timesteps, mean_kl, lw=2.5, color=color, linestyle=linestyle, label=label)
        ax.fill_between(timesteps, mean_kl - se_kl, mean_kl + se_kl, alpha=0.1, color=color)
    
    # Title with both train sizes if cross-size mode
    if train_size_2 is not None:
        title = f"Information Gap Analysis (N_train={train_size} vs {train_size_2})\nPairs: {n_pairs_done}/{n_pairs_total}"
        base = f"gap_analysis_N_train_{train_size}_vs_{train_size_2}_pairs_{n_pairs_total}_{tag}"
    else:
        title = f"Information Gap Analysis (N_train={train_size})\nPairs: {n_pairs_done}/{n_pairs_total}"
        base = f"gap_analysis_N_train_{train_size}_pairs_{n_pairs_total}_{tag}"
    
    ax.set_title(title)
    ax.set_xlabel("Denoising timestep t")
    ax.set_ylabel("KL Divergence (nats)")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_png = plots_dir / f"{base}.png"
    out_npz = plots_dir / f"{base}.npz"

    _atomic_savefig(fig, out_png, bbox_inches="tight")
    plt.close(fig)
    
    # Save raw data
    save_dict = {"timesteps": timesteps}
    for k, data in bp_gaps.items():
        save_dict[f"kl_bp0_vs_bp{k}"] = data
    for model_name, data in model_gaps.items():
        save_dict[f"kl_bp0_vs_{model_name}"] = data
    for model_name, epoch in epochs_used.items():
        save_dict[f"epoch_{model_name}"] = np.array([epoch])
        
    _atomic_savez(out_npz, **save_dict)
    logger.info("Saved gap analysis plot: %s", out_png)
    logger.info("Saved gap analysis data: %s", out_npz)


def main(args):
    """Entry point for score divergence along t (Exp 5 / Fig 3b).

    Computes KL(BP_0 || BP_k) and KL(BP_0 || NN) across diffusion timesteps
    for multiple training sizes, then produces gap-analysis plots.

    Args:
        args (argparse.Namespace): CLI arguments.
    """
    # Set GPU if available
    device = setup_device()
    
    # Parse factorized layers from comma-separated string (k values for BP approximation)
    # NOTE: for gap analysis, we compute KL(BP_0 || BP_k) for k in f_list (excluding 0)
    f_list_input = [int(x.strip()) for x in args.bp_factorized_layers.split(",")]
    # Separate: we need k=0 (exact BP) as reference, and k=1,2,3,4 for gaps
    k_values_for_gap = [k for k in f_list_input if k != 0]
    logger.info(f"Gap analysis: k values for BP approximation: {k_values_for_gap}")

    # Determine mode: multi-size (3 train sizes), cross-size (2), or same-size (1)
    multi_size_mode = args.train_size_3 is not None and args.train_size_3 > 0
    cross_size_mode = args.train_size_2 is not None and args.train_size_2 != args.train_size
    
    # Parse data_ids for multi-size mode
    if args.data_ids:
        data_ids = [int(x.strip()) for x in args.data_ids.split(",")]
    else:
        data_ids = None  # Will use default (all pick 0)
    
    if multi_size_mode:
        # Multi-size: compare 3 train sizes (e.g. 5k vs 12k vs 70k)
        train_sizes = [args.train_size, args.train_size_2, args.train_size_3]
        if data_ids is None:
            data_ids = [0, 0, 1]  # Default: 5k and 12k at pick 0, 70k at pick 1
        
        logger.info(f"Multi-size mode: comparing {train_sizes} at picks {data_ids}")
        models_by_size = discover_multi_size_models(
            args.base_path, train_sizes, data_ids, max_seeds=args.max_seeds
        )
        
        model_labels = {
            "Model1": f"Model_{train_sizes[0]//1000}k",
            "Model2": f"Model_{train_sizes[1]//1000}k",
            "Model3": f"Model_{train_sizes[2]//1000}k"
        }
        plots_dir_name = f"N_train_{train_sizes[0]}_vs_{train_sizes[1]}_vs_{train_sizes[2]}"
        pairs = None  # Not using pairs in multi-size mode
        
    elif cross_size_mode:
        # Cross-size: compare train_size (e.g. 5k) vs train_size_2 (e.g. 12k)
        logger.info(f"Cross-size mode: comparing {args.train_size} vs {args.train_size_2}")
        pairs = discover_cross_size_pairs(
            args.base_path, args.train_size, args.train_size_2, args.num_datasets,
            max_seeds=args.max_seeds
        )
        model_labels = {
            "Model1": f"Model_{args.train_size//1000}k",
            "Model2": f"Model_{args.train_size_2//1000}k"
        }
        all_dirs = []  # Not used in cross-size mode
        plots_dir_name = f"N_train_{args.train_size}_vs_{args.train_size_2}"
        models_by_size = None
    else:
        # Same-size: compare seeds within same train size
        all_dirs, pairs = discover_pairs(args.base_path, args.same_dataset, args.train_size, args.num_datasets)
        model_labels = {"Model1": "Model1", "Model2": "Model2"}
        plots_dir_name = f"N_train_{args.train_size}"
        models_by_size = None
    
    if multi_size_mode:
        # In multi-size mode, we have models_by_size instead of pairs
        total_models = sum(len(v) for v in models_by_size.values())
        logger.info(f"Discovered {total_models} models across {len(models_by_size)} train sizes")
        unique_paths = sorted({Path(p) for paths in models_by_size.values() for p in paths})
    else:
        logger.info("Discovered pairs: %d", len(pairs))
        for i, p in enumerate(pairs):
            logger.info(f"Pair {i}: {p[0]} <-> {p[1]}")
        unique_paths = sorted({Path(p) for pair in pairs for p in pair})

    _plots_base = Path(args.output_root) if args.output_root else Path("..") / "plots"
    plots_dir = (
        _plots_base
        / "exp5_gap_analysis"
        / plots_dir_name
        / f"N_test_{args.N_test}"
        / f"Same_{args.same_dataset}"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    # This is an experiment cache, it stores compiled model and associated parameters
    exp_cache: Dict[Path, dict] = {}
    # Used to store diffusion schedulers
    schedule_cache: Dict[Tuple[Any, ...], dict] = {}

    # List of unique model paths
    logger.info("Unique experiment folders: %d", len(unique_paths))

    # For all paths, instantiate corresponding models, compile them and cache them
    for p in tqdm(unique_paths, desc="Building experiments"):
        build_or_get_experiment(exp_cache, p, device, do_compile=(not args.no_compile))

    # Noised cache is used to store noised data 
    noised_cache: Dict[Tuple[Any, ...], torch.Tensor] = {}
    bp_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
    pred_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

    # Accumulators for gap analysis results
    bp_gaps_all: Dict[int, np.ndarray] = {}  # k -> KL(BP_0 || BP_k)
    model_gaps_all: Dict[str, np.ndarray] = {}  # "Model1"/"Model2"/"Model3" -> KL(BP_0 || Model)
    epochs_used: Dict[str, int] = {}  # track which epoch was evaluated per model
    
    pairs_done = 0

    ##### Multi-Size Mode #####
    if multi_size_mode:
        # In multi-size mode, compute BP once using the first experiment
        # Since all picks share same controlled setting structure, BP is identical
        first_path = unique_paths[0]
        ref_exp = exp_cache[first_path]
        
        params = ref_exp["params"]
        vocab_size = ref_exp["vocab_size"]
        seq_len = ref_exp["seq_len"]
        sequences_ref = ref_exp["sequences"]
        t_final = int(params["t_final"])
        reduced_length = int(params["reduced_length"])
        pick_i = int(params.get("pick_i_for_training", 0) or 0)
        s_val = float(params["s"])
        
        # Get schedule
        sched = get_schedule(schedule_cache, t_test=t_final, s=s_val, device=device)
        
        # Test indices (use sequences from reference experiment)
        # Test set starts after the training region for the first train size
        # The training region is: [pick_i * reduced_length, pick_i * reduced_length + train_size_from_path]
        # But train_size isn't in params - use reduced_length as the training region boundary
        li = pick_i * reduced_length
        ri = li + reduced_length  # Training region ends at reduced_length
        test_start = ri
        test_end = min(test_start + args.N_test, len(sequences_ref))
        test_idx = list(range(test_start, test_end))
        test_seqs = sequences_ref[test_idx].long()
        test_oh = torch.nn.functional.one_hot(test_seqs, num_classes=vocab_size).float().to(device)
        
        # Repeat for multiple reps like in pairs mode
        test_oh_rep = test_oh[:args.N_test].repeat(args.reps, 1, 1)
        
        # Get rho and k from reference experiment (same for all controlled setting data)
        rho = ref_exp["rho"]
        k_tree = ref_exp["k"]
        
        # Generate noised trajectory once
        t_max = max(1, int(args.t_denoising_ratio * t_final))
        x_test_noised = get_noised_stack(
            noised_cache, x0_one_hot=test_oh_rep, t_max=t_max,
            sched=sched, seed=args.seed,
        )
        batch = x_test_noised.shape[1]  # [t_max+1, batch, seq_len, vocab]
        
        # Compute BP once (for all k values including 0)
        logger.info("Computing BP once for all factorization levels...")
        bp_exact = get_bp_preds_trajectory(
            bp_cache, x_noised_stack=x_test_noised, t_max=t_max,
            rho=rho, k=k_tree, vocab_size=vocab_size,
            sched=sched, factorized_layers=0, bp_backend=args.bp_backend, device=device,
        )
        
        # Compute BP for k=1,2,3,4 and their KL gaps
        for k_val in k_values_for_gap:
            bp_k = get_bp_preds_trajectory(
                bp_cache, x_noised_stack=x_test_noised, t_max=t_max,
                rho=rho, k=k_tree, vocab_size=vocab_size,
                sched=sched, factorized_layers=k_val, bp_backend=args.bp_backend, device=device,
            )
            kl_gap = np.zeros((t_max, batch), dtype=np.float64)
            for idx in range(t_max):
                kl_gap[idx] = kl_divergence_batch(bp_exact[idx], bp_k[idx])
            bp_gaps_all[k_val] = kl_gap
        
        # Now compute model gaps for each train size
        model_name_map = {train_sizes[0]: "Model1", train_sizes[1]: "Model2", train_sizes[2]: "Model3"}
        
        for train_size in train_sizes:
            model_name = model_name_map[train_size]
            model_paths = models_by_size[train_size]
            
            if not model_paths:
                logger.warning(f"No models found for {train_size}, skipping")
                continue
            
            # Accumulate KL across seeds
            kl_accum = None
            n_seeds = 0
            
            for model_path in model_paths:
                exp = exp_cache[model_path]
                
                # Find epoch: use size-specific epoch for each model
                if model_name == "Model3":
                    epoch = args.which_epoch3
                elif model_name == "Model2":
                    epoch = args.which_epoch2
                else:
                    epoch = args.which_epoch1
                
                epochs_used[model_name] = epoch  # last one wins (for display)
                
                # Load checkpoint
                load_epoch_weights_once(model_path, exp_cache, epoch, device)
                
                # Get model predictions
                m_preds = denoise_trajectory_cached(
                    pred_cache, model_path=model_path, model=exp["model"],
                    x_noised_stack=x_test_noised, t_max=t_max, t_final=t_final,
                    sched=sched, device=device, split="test", epoch=epoch,
                )
                
                # Compute KL(BP_0 || Model)
                kl_model = np.zeros((t_max, batch), dtype=np.float64)
                for idx in range(t_max):
                    kl_model[idx] = kl_divergence_batch(bp_exact[idx], m_preds[idx])
                
                if kl_accum is None:
                    kl_accum = kl_model
                else:
                    kl_accum += kl_model
                n_seeds += 1
            
            # Average across seeds
            if n_seeds > 0:
                model_gaps_all[model_name] = kl_accum / n_seeds
                logger.info(f"Averaged {model_name} ({train_size}) over {n_seeds} seeds")
        
        pairs_done = 1  # For plot compatibility
        
        # Plot final result
        timesteps = np.arange(t_max, dtype=int)
        plot_gap_analysis(
            plots_dir=plots_dir,
            timesteps=timesteps,
            bp_gaps=bp_gaps_all,
            model_gaps=model_gaps_all,
            n_pairs_done=1,
            n_pairs_total=1,
            epochs_used=epochs_used,
            train_size=train_sizes[0],
            train_size_2=train_sizes[1],
            model_labels=model_labels,
            tag="latest",
        )
        
        logger.info("Done. Multi-size gap analysis complete.")
        return

    # Pairs mode(cross-size or same-size)
    # Loop over path pairs
    for (path1, path2) in tqdm(pairs, desc="Pairs", leave=True):
        path1 = Path(path1)
        path2 = Path(path2)

        # Retrieve experiments given paths
        e1 = exp_cache[path1]
        e2 = exp_cache[path2]
        
        # Determine epochs to use
        epoch1 = args.which_epoch1
        epoch2 = args.which_epoch2
        
        # Track epochs used
        epochs_used["Model1"] = epoch1
        epochs_used["Model2"] = epoch2
        
        # Load checkpoints for both models
        load_epoch_weights_once(path1, exp_cache, epoch1, device)
        load_epoch_weights_once(path2, exp_cache, epoch2, device)
        
        # Get parameters from e1
        params1 = e1["params"]
        sequences = e1["sequences"]
        vocab_size = int(e1["vocab_size"])
        seq_len = int(e1["seq_len"])
        rho, k_tree = e1["rho"], e1["k"]

        num_seqs = sequences.shape[0]

        # Usual training set extraction logic
        reduced_length1 = int(params1["reduced_length"])
        pick_i1 = int(params1.get("pick_i_for_training", 0) or 0)
        
        params2 = e2["params"]
        reduced_length2 = int(params2["reduced_length"])
        pick_i2 = int(params2.get("pick_i_for_training", 0) or 0)
        
        train_idx1, used_pick_i1 = train_idx_from_pick_i(
            num_seqs=num_seqs,
            reduced_length=reduced_length1,
            pick_i=pick_i1,
            device=sequences.device,
        )
        train_idx2, used_pick_i2 = train_idx_from_pick_i(
            num_seqs=num_seqs,
            reduced_length=reduced_length2,
            pick_i=pick_i2,
            device=sequences.device,
        )

        # Shared test set disjoint from BOTH train blocks (index+content)
        test_idx = build_shared_test_idx_disjoint_by_content(
            sequences=sequences,
            train_idx1=train_idx1,
            train_idx2=train_idx2,
            N_test=args.N_test,
            seed=args.seed,
        )

        # Select test sequences
        test_seqs = sequences[test_idx]

        # One-hot + reps
        test_oh = torch.nn.functional.one_hot(test_seqs.long(), num_classes=vocab_size).float().to(device)
        test_oh = test_oh[: args.N_test].repeat(args.reps, 1, 1)

        # Schedule
        t_final = int(params1["t_final"])
        t_test = t_final
        t_max = int(max(1, int(args.t_denoising_ratio * t_final)))

        sched = get_schedule(
            schedule_cache,
            t_test=t_test,
            s=float(params1["s"]),
            device=device,
        )

        # Noised trajectories (test set)
        x_test_noised = get_noised_stack(
            noised_cache,
            x0_one_hot=test_oh,
            t_max=t_max,
            sched=sched,
            seed=args.seed + 17 * (pairs_done + 1),
        )

        batch = x_test_noised.shape[1]
        
        # 1. Compute exact BP (k=0) as reference
        bp_exact = get_bp_preds_trajectory(
            bp_cache,
            x_noised_stack=x_test_noised,
            t_max=t_max,
            rho=rho,
            k=k_tree,
            vocab_size=vocab_size,
            sched=sched,
            device=device,
            bp_backend=args.bp_backend,
            factorized_layers=0,  # Exact BP
        )
        
        # 2. Compute BP approximation gaps (k=1,2,3,4)
        bp_gaps_pair: Dict[int, np.ndarray] = {}
        
        for f_k in k_values_for_gap:
            bp_k = get_bp_preds_trajectory(
                bp_cache,
                x_noised_stack=x_test_noised,
                t_max=t_max,
                rho=rho,
                k=k_tree,
                vocab_size=vocab_size,
                sched=sched,
                device=device,
                bp_backend=args.bp_backend,
                factorized_layers=f_k,
            )
            
            # Compute KL(BP_0 || BP_k), exact BP is reference
            kl_traj = np.zeros((t_max, batch), dtype=np.float64)
            for idx in range(t_max):
                kl_traj[idx] = kl_divergence_batch(bp_exact[idx], bp_k[idx])
            
            bp_gaps_pair[f_k] = kl_traj
        
        # 3. Compute model gaps
        model_gaps_pair: Dict[str, np.ndarray] = {}
        
        # Model 1 predictions
        m1_preds = denoise_trajectory_cached(
            pred_cache,
            model_path=path1,
            model=e1["model"],
            x_noised_stack=x_test_noised,
            t_max=t_max,
            t_final=t_final,
            sched=sched,
            device=device,
            split="test",
            epoch=epoch1,
        )
        
        # KL(BP_0 vs Model1)
        kl_m1 = np.zeros((t_max, batch), dtype=np.float64)
        for idx in range(t_max):
            kl_m1[idx] = kl_divergence_batch(bp_exact[idx], m1_preds[idx])
        model_gaps_pair["Model1"] = kl_m1
        
        # Model 2 predictions
        m2_preds = denoise_trajectory_cached(
            pred_cache,
            model_path=path2,
            model=e2["model"],
            x_noised_stack=x_test_noised,
            t_max=t_max,
            t_final=t_final,
            sched=sched,
            device=device,
            split="test",
            epoch=epoch2,
        )
        
        # KL(BP_0 vs Model2)
        kl_m2 = np.zeros((t_max, batch), dtype=np.float64)
        for idx in range(t_max):
            kl_m2[idx] = kl_divergence_batch(bp_exact[idx], m2_preds[idx])
        model_gaps_pair["Model2"] = kl_m2

        # Update running averages
        pairs_done += 1
        n = float(pairs_done)
        
        # Update BP gaps
        for k_val, kl_pair in bp_gaps_pair.items():
            if k_val not in bp_gaps_all:
                bp_gaps_all[k_val] = kl_pair
            else:
                bp_gaps_all[k_val] += (kl_pair - bp_gaps_all[k_val]) / n
        
        # Update model gaps
        for model_name, kl_pair in model_gaps_pair.items():
            if model_name not in model_gaps_all:
                model_gaps_all[model_name] = kl_pair
            else:
                model_gaps_all[model_name] += (kl_pair - model_gaps_all[model_name]) / n

        # Plot current state
        timesteps = np.arange(t_max, dtype=int)
        plot_gap_analysis(
            plots_dir=plots_dir,
            timesteps=timesteps,
            bp_gaps=bp_gaps_all,
            model_gaps=model_gaps_all,
            n_pairs_done=pairs_done,
            n_pairs_total=len(pairs),
            epochs_used=epochs_used,
            train_size=args.train_size,
            train_size_2=args.train_size_2,
            model_labels=model_labels,
            tag="latest",
        )

    if pairs_done == 0:
        logger.error("No pairs processed successfully (all skipped).")
        return

    logger.info("Done. Processed %d/%d pairs. Outputs in %s", pairs_done, len(pairs), plots_dir)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument(
        "--base-path",
        type=str,
        default="../results_transformer_pick_{}_for_training",
        help="Base model directory with {} placeholder for data_id.",
    )
    p.add_argument("--num-datasets", type=int, default=15)
    p.add_argument("--same-dataset", type=int, default=0)
    p.add_argument("--train-size", dest="train_size", type=int, default=12000)
    p.add_argument("--train-size-2", dest="train_size_2", type=int, default=5000,
                   help="Second train size for cross-size comparison (e.g., compare 5k vs 12k on same plot).")
    p.add_argument("--train-size-3", dest="train_size_3", type=int, default=70000,
                   help="Third train size (e.g., 70000 for 70k model).")

    p.add_argument("--N-test", dest="N_test", type=int, default=2000)

    p.add_argument("--which-epoch1", type=int, required=True,
                   help="Epoch for --train-size model. Should be the test-loss "
                        "minimum (find it via nn_divergence.py or "
                        "loss_decomposition.py).")
    p.add_argument("--which-epoch2", type=int, required=True,
                   help="Epoch for --train-size-2 model. Same criterion as "
                        "--which-epoch1.")
    p.add_argument("--which-epoch3", type=int, required=True,
                   help="Epoch for --train-size-3 model. Same criterion as "
                        "--which-epoch1.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--t-denoising-ratio", type=float, default=1.0, help="Ratio of t_final to use for denoising trajectory (default: 1.0, meaning full length).")

    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--bp-backend", type=str, choices=['torch', 'numpy'], default='torch',
                   help="Backend for BP computation: 'torch' (GPU batched) or 'numpy' (CPU serial). Default: torch")
    
    p.add_argument("--bp-factorized-layers", type=str, default="0,1,2,3,4", 
                   help="Comma-separated list of factorization levels to analyze (e.g. '0,1,2'). 0=Exact BP.")
    
    p.add_argument("--max-seeds", dest="max_seeds", type=int, default=5,
                   help="Maximum number of seeds to average over in cross-size mode (default: 5).")
    p.add_argument("--data-ids", dest="data_ids", type=str, default=None,
                   help="Comma-separated data IDs (picks) for each train size. E.g., '0,0,1' for 5k@pick0, 12k@pick0, 70k@pick1.")
    p.add_argument("--output-root", type=str, default=None,
                   help="Root dir for output plots (default: ../plots)")
    
    args = p.parse_args()
    main(args)
