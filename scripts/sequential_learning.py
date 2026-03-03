"""
sequential_learning.py - Sequential learning and sample-split score divergence vs training epoch
Referred to as exp6 in output directory and plotting functions.
Reproduces Fig. 1(b) (n=12k) and Fig. 6 (n=5k, n=70k).
=================================================================================

Tracks the evolution of model-oracle divergence (KL to BP_k for k=0,...,ell) and inter-model
divergence (sample-split KL) across training epochs at a fixed critical
diffusion time t/T ~ 0.15 (where the KL divergence peaks, see Sec. 4.2).

The sample-split KL minimum identifies the onset of biased generalization:
it occurs before the model-oracle KL minimum (proxy for the test-loss
minimum), confirming that bias emerges while the model is still improving
toward the exact score.

Procedure:
1. Model pairing (--same-dataset):
   - Same dataset: measures seed variance.
   - Disjoint dataset: sample-split analysis measuring data-dependent bias.
2. Fixed time evaluation: at timestep t defined by --t-denoising-ratio,
   noise test/train samples and compute posterior mean predictions.
3. At the fixed timestep t, compute three KL divergences:
   - D_KL(BP_k || model): model-oracle divergence for each BP_k level.
   - D_KL(model_A || model_B): sample-split inter-model disagreement.
4. Repeat for all available training checkpoints.

Outputs:
- Evolution plots of mean KL divergences vs training epochs.
- Separate panels for test data (generalization) and training data (fitting).
- NPZ files containing the raw trajectories.

Interpretation:
- Sample-split KL minimizes first: models have converged to a shared
  approximation (still data-independent at this point).
- Model-oracle KL minimizes later: the model reaches its closest
  approximation to the true posterior.
- Between these two minima lies the biased generalization phase.

Usage (run from scripts/ directory)::

    # within-split stability (n=12k)
    python sequential_learning.py \\
        --output-root ../plots/ \\
        --base-path1 "<model_root>" \\
        --train-size 12000 --same-dataset 1 --N-test 3000 --reps 5

    # cross-split sample-split analysis (n=12k, Fig. 1(b))
    python sequential_learning.py \\
        --output-root ../plots/ \\
        --base-path1 "<model_root>" \\
        --train-size 12000 --same-dataset 0 --N-test 3000 --reps 5

    # for Fig. 6, change --train-size to 5000 or 70000
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
import itertools
import math
import tempfile
import os
import json
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.diffusion import forward_process, backward_process, backward_process_gt
from modules.bp_torch import run_BP_diffusion_torch
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


def mean_and_se(x: np.ndarray):
    x = np.asarray(x)
    n = x.size
    if n <= 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))


def discover_pairs(base_path_fmt: str, same_dataset: int, train_size: int, num_datasets: int = 14):
    """
    Leaf dirs containing 'train_size', expecting 2 per dataset (two seeds). Computes pairs once.
    """
    all_dirs = []
    for data_id in range(num_datasets):
        root = Path(base_path_fmt.format(data_id))
        if not root.exists():
            print(f"Warning: base path missing: {root}")
            continue

        dirs_with_train_size = [
            d for d in root.rglob("*") if d.is_dir() and ("_" + str(train_size)) in str(d)
        ]
        leaf_dirs = [d for d in dirs_with_train_size if not any(x.is_dir() for x in d.iterdir())]

        if len(leaf_dirs) > 2:
            all_dirs.append(leaf_dirs[-2:])
        elif len(leaf_dirs) == 2:
            all_dirs.append(leaf_dirs)
        else:
            continue

    if len(all_dirs) == 0:
        raise RuntimeError("No eligible experiment directories found.")

    pairs_dir = np.array(all_dirs, dtype=object)

    if same_dataset == 1:
        # Row-wise: compare the two seeds within each dataset
        pairs = [(pairs_dir[i, 0], pairs_dir[i, 1]) for i in range(pairs_dir.shape[0])]
    else:
        # Across datasets: each dataset-pair once, all seed combinations
        pairs = [
            (pairs_dir[i, s1], pairs_dir[j, s2])
            for (i, j) in itertools.combinations(range(len(all_dirs)), 2)
            for s1 in range(2)
            for s2 in range(2)
        ]

    return all_dirs, pairs


def safe_block_indices(
    *, num_seqs: int, reduced_length: int, train_block_i: int, test_block_i: int = 15, device=None
):
    """
    Train block is train_block_i, Test block is test_block_i.
    Fallback deterministically if test block invalid or equals train block.
    """
    max_blocks = num_seqs // reduced_length
    if max_blocks <= 1:
        raise ValueError(f"Not enough sequences ({num_seqs}) for reduced_length={reduced_length}.")

    def block_to_idx(bi: int) -> torch.Tensor:
        left = bi * reduced_length
        right = left + reduced_length
        if left < 0 or right > num_seqs:
            raise ValueError(
                f"Block {bi} out of bounds for num_seqs={num_seqs}, reduced_length={reduced_length}"
            )
        return torch.arange(left, right, device=device)

    train_block_i = int(train_block_i)
    test_block_i = int(test_block_i)

    train_idx = block_to_idx(train_block_i)

    if (0 <= test_block_i < max_blocks) and (test_block_i != train_block_i):
        return train_idx, block_to_idx(test_block_i), test_block_i

    fb = 0 if train_block_i != 0 else 1
    if fb >= max_blocks:
        raise ValueError("Could not find a fallback test block.")
    return train_idx, block_to_idx(fb), fb


def get_schedule(schedule_cache: dict, *, t_test: int, s: float, device):
    """
    Cache diffusion schedule (alpha_bars/alphas) since it's reused heavily.
    """
    key = (int(t_test), float(s), device.type)
    if key in schedule_cache:
        return schedule_cache[key]

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
    """
    Cache by experiment directory path:
      params + data + model (optionally compiled) built once.
    """
    path = Path(path)
    if path in exp_cache:
        return exp_cache[path]

    params = load_params(path / "full_params.json")

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


def load_epoch_weights_once(unique_paths, exp_cache, epoch: int, device):
    """
    Load checkpoints once per unique model path per epoch.
    """
    for p in unique_paths:
        exp = exp_cache[p]
        ckpt = p / f"test_model_script_epoch_{epoch}.pt"
        if (not ckpt.exists()) or (
            not load_checkpoint(exp["model"], str(ckpt), device, training=False)
        ):
            raise RuntimeError(f"Checkpoint load failed: {ckpt}")
        exp["model"].eval()


def dataset_key(exp: dict) -> str:
    """
    BP is cached per dataset. We treat dataset identity as params['data_path'].
    """
    return str(exp["params"]["data_path"])


# ----------------------------- BP (FACTORISED) CACHING -----------------------------


def _compute_bp_factorized_np(
    x_noised_np: np.ndarray,
    *,
    t_noise: int,
    bp_params: tuple,
    sched: dict,
    factorized_layers: int,
) -> np.ndarray:
    """
    Compute BP marginals with given factorized_layers using numpy path.
    """
    rho, k, vocab_size = bp_params
    _, bp_x0 = backward_process_gt(
        x_noised_np,
        int(t_noise),
        (rho, k, vocab_size),
        device=None,  # numpy path
        alpha_bars=sched["alpha_bars_np"],
        alphas=sched["alphas_np"],
        fix_noise=False,  # KL is on x0_hat score predictions; posterior noise does not enter the metric
        factorized_layers=int(factorized_layers),
    )
    if isinstance(bp_x0, torch.Tensor):
        bp_x0 = bp_x0.detach().cpu().numpy()
    return bp_x0


def _compute_bp_factorized_torch(
    x_noised: torch.Tensor,
    *,
    t_noise: int,
    bp_params: tuple,
    sched: dict,
    factorized_layers: int,
    device,
) -> np.ndarray:
    """
    Compute BP marginals with given factorized_layers using PyTorch path.
    
    Returns numpy array for compatibility with kl_divergence_batch.
    """
    # bp_params is (M, l, q) where M=rho (transition matrix), l=k (depth), q=vocab_size
    M, l, q = bp_params
    
    # Convert M to tensor if numpy
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).float().to(device)
    else:
        M = M.to(device).float()
    
    # Compute field intensity (same as numpy path in backward_process_gt)
    alpha_bars = sched["alpha_bars"]
    if isinstance(alpha_bars, np.ndarray):
        field_intensity = np.sqrt(alpha_bars[t_noise - 1]) / (1 - alpha_bars[t_noise - 1])
        field_intensity = torch.tensor(field_intensity, device=device, dtype=torch.float32)
    else:
        field_intensity = torch.sqrt(alpha_bars[t_noise - 1]) / (1 - alpha_bars[t_noise - 1])
        field_intensity = field_intensity.to(device)
    
    # x_noised is (B, seq_len, vocab_size) - apply field intensity
    field = x_noised * field_intensity
    
    # Run torch BP
    bp_marginals = run_BP_diffusion_torch(
        M, l, q, field, factorized_layers=int(factorized_layers)
    )
    
    # Convert to numpy for KL computation
    return bp_marginals.detach().cpu().numpy()


def get_cached_bp_factorized(
    pack: dict, factorized_layers: int, backend: str = "numpy"
) -> np.ndarray:
    """
    Cache BP results for all factorized layers inside the pack:
      pack["bp_preds_factorized"][f] = BP(f) prediction.

    Args:
        pack: Data pack containing x_noised and BP parameters
        factorized_layers: Number of factorized layers (0 = exact)
        backend: 'torch' or 'numpy'
    """
    f = int(factorized_layers)
    cache_key = f"bp_preds_factorized_{backend}"
    cache = pack.setdefault(cache_key, {})
    if f in cache:
        return cache[f]

    if backend == "torch":
        # Use torch path - x_noised should already be on GPU
        bp = _compute_bp_factorized_torch(
            pack["x_noised"],
            t_noise=int(pack["t_noise"]),
            bp_params=pack["bp_params"],
            sched=pack["schedule"],
            factorized_layers=f,
            device=pack["x_noised"].device,
        )
    else:
        # Use numpy path
        if "x_noised_np" not in pack or pack["x_noised_np"] is None:
            pack["x_noised_np"] = pack["x_noised"].detach().cpu().numpy()
        bp = _compute_bp_factorized_np(
            pack["x_noised_np"],
            t_noise=int(pack["t_noise"]),
            bp_params=pack["bp_params"],
            sched=pack["schedule"],
            factorized_layers=f,
        )
    
    cache[f] = bp
    return bp



# ----------------------------- BP PACKS (TEST/TRAIN) -----------------------------


def get_bp_test_pack(
    *,
    eval_exp: dict,
    device,
    schedule_cache: dict,
    bp_test_cache: dict,
    t_final_test_override: Optional[int],
    t_denoising_ratio: float,
    N_test: int,
    reps: int,
    test_block_i: int,
):
    """
    BP(TEST) depends only on dataset/config; cached NOT per epoch.
    """
    params = eval_exp["params"]
    sequences = eval_exp["sequences"]
    vocab_size = eval_exp["vocab_size"]
    rho, k = eval_exp["rho"], eval_exp["k"]

    dkey = dataset_key(eval_exp)
    reduced_length = int(params["reduced_length"])
    pick_i = int(params.get("pick_i_for_training", 0) or 0)

    t_final = int(params["t_final"])
    t_test = int(t_final_test_override) if t_final_test_override is not None else t_final
    t_noise = int(t_denoising_ratio * t_final)

    sched = get_schedule(
        schedule_cache,
        t_test=t_test,
        s=float(params["s"]),
        device=device,
    )

    key = (
        dkey,
        int(t_test),
        float(params["s"]),
        int(t_noise),
        int(N_test),
        int(reps),
        int(test_block_i),
    )
    if key in bp_test_cache:
        return bp_test_cache[key]

    _, test_idx, used_test_block = safe_block_indices(
        num_seqs=sequences.shape[0],
        reduced_length=reduced_length,
        train_block_i=pick_i,
        test_block_i=test_block_i,
        device=sequences.device,
    )
    if int(used_test_block) != int(test_block_i):
        logger.warning(
            "Requested test_block_i=%d invalid/overlapping; using fallback test_block=%d for dataset=%s",
            int(test_block_i),
            int(used_test_block),
            dkey,
        )

    test_seqs = sequences[test_idx]
    test_oh = torch.nn.functional.one_hot(test_seqs.long(), num_classes=vocab_size).float().to(device)
    x_test_noised = forward_process(
        test_oh[:N_test].repeat(reps, 1, 1),
        torch.tensor(t_noise, device=device, dtype=torch.long),
        alpha_bars=sched["alpha_bars"],
    )

    pack = dict(
        split="test",
        dataset_key=dkey,
        schedule=sched,
        t_final=t_final,
        t_test=t_test,
        t_noise=t_noise,
        test_block_i=int(test_block_i),
        used_test_block=int(used_test_block),
        x_test_noised=x_test_noised,
        x_noised=x_test_noised,  # canonical
        x_noised_np=None,
        bp_params=(rho, k, vocab_size),
        vocab_size=int(vocab_size),
        seq_len=int(eval_exp["seq_len"]),
        tree_depth=int(k),
        bp_preds_factorized={},
    )

    pack["bp_preds_test_k0"] = get_cached_bp_factorized(pack, 0, backend="numpy")  # Default for setup

    bp_test_cache[key] = pack
    return pack


def get_bp_train_pack_for_model1(
    *,
    eval_exp: dict,
    device,
    schedule_cache: dict,
    bp_train_cache: dict,
    t_final_test_override: Optional[int],
    t_denoising_ratio: float,
    N_test: int,
    reps: int,
    test_block_i: int,
    print_overlap_once: bool,
):
    """
    BP(TRAIN) for model1 block; cached NOT per epoch.
    """
    params = eval_exp["params"]
    sequences = eval_exp["sequences"]
    vocab_size = eval_exp["vocab_size"]
    rho, k = eval_exp["rho"], eval_exp["k"]

    dkey = dataset_key(eval_exp)
    reduced_length = int(params["reduced_length"])
    pick_i = int(params.get("pick_i_for_training", 0) or 0)

    t_final = int(params["t_final"])
    t_test = int(t_final_test_override) if t_final_test_override is not None else t_final
    t_noise = int(t_denoising_ratio * t_final)

    sched = get_schedule(
        schedule_cache,
        t_test=t_test,
        s=float(params["s"]),
        device=device,
    )

    key = (
        dkey,
        int(pick_i),
        int(reduced_length),
        int(t_test),
        float(params["s"]),
        int(t_noise),
        int(N_test),
        int(reps),
        int(test_block_i),
    )
    if key in bp_train_cache:
        return bp_train_cache[key]

    train_idx, test_idx, used_test_block = safe_block_indices(
        num_seqs=sequences.shape[0],
        reduced_length=reduced_length,
        train_block_i=pick_i,
        test_block_i=test_block_i,
        device=sequences.device,
    )
    if int(used_test_block) != int(test_block_i):
        logger.warning(
            "Requested test_block_i=%d invalid/overlapping; using fallback test_block=%d for dataset=%s",
            int(test_block_i),
            int(used_test_block),
            dkey,
        )

    train_seqs = sequences[train_idx]
    test_seqs = sequences[test_idx]

    if print_overlap_once:
        overlap = (
            (train_seqs[:, None, :] == test_seqs[None, :, :]).all(-1).any(0).float().mean().item()
        )
        logger.info(
            "Dataset=%s | pick_i(model1)=%d | test_block=%d | train/test overlap=%.4f",
            dkey,
            pick_i,
            used_test_block,
            overlap,
        )

    train_oh = torch.nn.functional.one_hot(train_seqs.long(), num_classes=vocab_size).float().to(device)
    x_train_noised = forward_process(
        train_oh[:N_test].repeat(reps, 1, 1),
        torch.tensor(t_noise, device=device, dtype=torch.long),
        alpha_bars=sched["alpha_bars"],
    )

    pack = dict(
        split="train",
        dataset_key=dkey,
        pick_i=int(pick_i),
        reduced_length=int(reduced_length),
        schedule=sched,
        t_final=t_final,
        t_test=t_test,
        t_noise=t_noise,
        used_test_block=int(used_test_block),
        x_train_noised=x_train_noised,
        x_noised=x_train_noised,  # canonical
        x_noised_np=None,
        bp_params=(rho, k, vocab_size),
        vocab_size=int(vocab_size),
        seq_len=int(eval_exp["seq_len"]),
        tree_depth=int(k),
        bp_preds_factorized={},
    )

    pack["bp_preds_train_k0"] = get_cached_bp_factorized(pack, 0, backend="numpy")  # Default for setup

    bp_train_cache[key] = pack
    return pack


# ------------------------- MODEL PREDICTION -------------------------


def denoise_cached(
    *,
    pred_cache_epoch: dict,
    model_path: Path,
    model,
    x_noised: torch.Tensor,
    split: str,
    dataset_key_str: str,
    t_noise: int,
    t_final: int,
    sched: dict,
    pick_i: Optional[int],
    device,
):
    """
    Cache model predictions within an epoch.
    """
    key = (
        str(model_path),
        dataset_key_str,
        split,
        int(t_noise),
        int(t_final),
        int(pick_i) if pick_i is not None else None,
    )
    if key in pred_cache_epoch:
        return pred_cache_epoch[key]

    with torch.inference_mode():
        _, x0_hat = backward_process(
            x_noised,
            int(t_noise),
            int(t_final),
            model,
            device,
            alpha_bars=sched["alpha_bars"],
            alphas=sched["alphas"],
            clamp=False,
            temperature=None,
            fix_noise=False,  # KL is on x0_hat score predictions; posterior noise does not enter the metric
        )
    out = x0_hat.detach().cpu().numpy()
    pred_cache_epoch[key] = out
    return out


# ------------------------------ ATOMIC SAVES ------------------------------


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


def save_summary_json(out_path: Path, data: dict):
    """
    Save curve data to JSON (numpy -> lists).
    """

    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def convert_structure(d):
        if isinstance(d, dict):
            return {k: convert_structure(v) for k, v in d.items()}
        if isinstance(d, list):
            return [convert_structure(i) for i in d]
        return to_list(d)

    clean_data = convert_structure(data)
    with open(out_path, "w") as f:
        json.dump(clean_data, f, indent=2)


# ------------------------------ PLOTTING ------------------------------


def plot_and_save(
    *,
    args_obj,
    plots_dir: Path,
    epochs_plot,
    title_time: int,
    t_denoising_ratio: float,
    reduced_length: int,
    reps: int,
    N_test: int,
    are_same: bool,
    has_train: bool,
    # legacy
    kl_bp_m1_test,
    kl_bp_m1_test_se,
    kl_bp_m2_test,
    kl_bp_m2_test_se,
    kl_m1_m2_test,
    kl_m1_m2_test_se,
    kl_bp_m1_train=None,
    kl_bp_m1_train_se=None,
    kl_bp_m2_train=None,
    kl_bp_m2_train_se=None,
    kl_m1_m2_train=None,
    kl_m1_m2_train_se=None,
    # factorized dicts
    kl_bp_f_m1_test: Optional[Dict[int, List[float]]] = None,
    kl_bp_f_m1_test_se: Optional[Dict[int, List[float]]] = None,
    kl_bp_f_m2_test: Optional[Dict[int, List[float]]] = None,
    kl_bp_f_m2_test_se: Optional[Dict[int, List[float]]] = None,
    kl_bp_f_m1_train: Optional[Dict[int, List[float]]] = None,
    kl_bp_f_m1_train_se: Optional[Dict[int, List[float]]] = None,
    kl_bp_f_m2_train: Optional[Dict[int, List[float]]] = None,
    kl_bp_f_m2_train_se: Optional[Dict[int, List[float]]] = None,
    tag: str = "latest",
    job_id: str = None,
):
    kl_bp_f_m1_test = kl_bp_f_m1_test or {}
    kl_bp_f_m1_test_se = kl_bp_f_m1_test_se or {}
    kl_bp_f_m2_test = kl_bp_f_m2_test or {}
    kl_bp_f_m2_test_se = kl_bp_f_m2_test_se or {}
    kl_bp_f_m1_train = kl_bp_f_m1_train or {}
    kl_bp_f_m1_train_se = kl_bp_f_m1_train_se or {}
    kl_bp_f_m2_train = kl_bp_f_m2_train or {}
    kl_bp_f_m2_train_se = kl_bp_f_m2_train_se or {}

    c_m1_m2 = "#2ca02c"
    colors_f = ["#d62728", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]

    if has_train:
        fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    else:
        fig, ax_test = plt.subplots(figsize=(8, 6))
        ax_train = None

    def panel(ax, epochs, c_vals, c_se, f_m1_dict, f_m1_se_dict, f_m2_dict, f_m2_se_dict, split):
        epochs = np.asarray(epochs, dtype=float)

        ax2 = ax.twinx()
        c_vals = np.asarray(c_vals, dtype=float)
        c_se = np.asarray(c_se, dtype=float)

        ax2.plot(
            epochs,
            c_vals,
            label=f"KL(M1||M2) {split}",
            linewidth=3.0,
            marker="^",
            color=c_m1_m2,
            linestyle="-",
        )
        ax2.fill_between(epochs, c_vals - c_se, c_vals + c_se, alpha=0.1, color=c_m1_m2)
        ax2.set_ylabel("KL(M1 || M2)", color=c_m1_m2, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=c_m1_m2)
        ax2.set_yscale("log")

        all_fs = sorted(set(f_m1_dict.keys()) | set(f_m2_dict.keys()))
        for f in all_fs:
            col = colors_f[f % len(colors_f)]

            if f in f_m1_dict:
                vals = np.asarray(f_m1_dict[f], dtype=float)
                errs = np.asarray(f_m1_se_dict[f], dtype=float)
                ax.plot(epochs, vals, label=f"BP(f={f}) || M1", linestyle="-", color=col, linewidth=2.0, alpha=0.8)
                if f == 0:
                    ax.fill_between(epochs, vals - errs, vals + errs, alpha=0.1, color=col)

            if f in f_m2_dict:
                vals = np.asarray(f_m2_dict[f], dtype=float)
                ax.plot(epochs, vals, label=f"BP(f={f}) || M2", linestyle="--", color=col, linewidth=2.0, alpha=0.8)

        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("KL(BP || M)", fontsize=12)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"KL divergences ({split})", fontsize=14)
        ax.grid(True, linestyle=":", alpha=0.7)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, ncol=3)

    panel(
        ax_test,
        epochs_plot,
        kl_m1_m2_test,
        kl_m1_m2_test_se,
        kl_bp_f_m1_test,
        kl_bp_f_m1_test_se,
        kl_bp_f_m2_test,
        kl_bp_f_m2_test_se,
        "test",
    )

    if has_train and ax_train is not None:
        panel(
            ax_train,
            epochs_plot,
            kl_m1_m2_train,
            kl_m1_m2_train_se,
            kl_bp_f_m1_train,
            kl_bp_f_m1_train_se,
            kl_bp_f_m2_train,
            kl_bp_f_m2_train_se,
            "train",
        )

    fig.suptitle(
        f"KL with Factorized BP at time {title_time} (ratio={t_denoising_ratio})",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    base = (
        f"KL_FACT_train_size_{reduced_length}_reps_{reps}_Ntest_{N_test}_"
        f"ratio_{t_denoising_ratio}_are_same_{are_same}_{tag}"
    )

    out_png = plots_dir / f"{base}.png"
    out_npz = plots_dir / f"{base}.npz"
    out_json = plots_dir / f"{base}.json"

    if job_id:
        out_png = plots_dir / f"{base}_{job_id}.png"
        out_npz = plots_dir / f"{base}_{job_id}.npz"
        out_json = plots_dir / f"{base}_{job_id}.json"

    _atomic_savefig(fig, out_png, bbox_inches="tight")
    plt.close(fig)

    save = {
        "args": {k: str(v) for k, v in vars(args_obj).items()},
        "epochs": np.asarray(epochs_plot),
        "kl_m1_m2_test": np.asarray(kl_m1_m2_test),
        "kl_m1_m2_test_se": np.asarray(kl_m1_m2_test_se),
    }

    for f, vals in kl_bp_f_m1_test.items():
        save[f"kl_bp_f{f}_m1_test"] = np.asarray(vals)
        save[f"kl_bp_f{f}_m1_test_se"] = np.asarray(kl_bp_f_m1_test_se[f])
    for f, vals in kl_bp_f_m2_test.items():
        save[f"kl_bp_f{f}_m2_test"] = np.asarray(vals)
        save[f"kl_bp_f{f}_m2_test_se"] = np.asarray(kl_bp_f_m2_test_se[f])

    if has_train:
        save["kl_m1_m2_train"] = np.asarray(kl_m1_m2_train)
        save["kl_m1_m2_train_se"] = np.asarray(kl_m1_m2_train_se)
        for f, vals in kl_bp_f_m1_train.items():
            save[f"kl_bp_f{f}_m1_train"] = np.asarray(vals)
            save[f"kl_bp_f{f}_m1_train_se"] = np.asarray(kl_bp_f_m1_train_se[f])
        for f, vals in kl_bp_f_m2_train.items():
            save[f"kl_bp_f{f}_m2_train"] = np.asarray(vals)
            save[f"kl_bp_f{f}_m2_train_se"] = np.asarray(kl_bp_f_m2_train_se[f])  # FIXED

    _atomic_savez(out_npz, **save)
    save_summary_json(out_json, save)

    logger.info("Saved plot: %s", out_png)
    logger.info("Saved data: %s", out_npz)
    logger.info("Saved json: %s", out_json)


# ------------------------------ MAIN ------------------------------


def main(args):
    """Entry point for the sequential-learning experiment (Exp 6 / Fig 1b, 6).

    Compares two models (trained on different dataset sizes) by computing
    KL divergences between their scores and exact/factorised BP scores
    across epochs.

    Args:
        args (argparse.Namespace): CLI arguments.
    """
    device = setup_device()
    has_train = args.has_train != 0

    try:
        f_list = [int(x) for x in args.bp_factorized_layers.split(",")]
    except ValueError:
        raise ValueError(f"Invalid format for --bp-factorized-layers: {args.bp_factorized_layers}")

    if args.models_root:
        if not (args.subpath1 and args.subpath2):
            raise ValueError("--models-root requires --subpath1 and --subpath2")
        root = Path(args.models_root)
        path1 = root / args.subpath1
        path2 = root / args.subpath2
        if not path1.exists():
            raise FileNotFoundError(f"Missing M1: {path1}")
        if not path2.exists():
            raise FileNotFoundError(f"Missing M2: {path2}")

        pairs = [(path1, path2)]
        all_dirs = [[path1, path2]]
        are_same = False

        logger.info("Explicit mode selected.")
        logger.info("M1: %s", path1)
        logger.info("M2: %s", path2)
    else:
        base_path1 = args.base_path1 + "results_transformer_pick_{}_for_training/"
        all_dirs, pairs = discover_pairs(base_path1, args.same_dataset, args.train_size)
        are_same = args.same_dataset == 1
        logger.info("Pairs identified via discovery: %d", len(pairs))

    _plots_base = Path(args.output_root) if args.output_root else Path("..") / "plots"
    plots_dir = (
        _plots_base
        / "exp6_factorized"
        / f"N_train_{args.train_size}"
        / f"N_test_{args.N_test}"
        / f"ratio_{args.t_denoising_ratio}"
        / f"backend_{args.bp_backend}"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    exp_cache: Dict[Path, dict] = {}
    schedule_cache: Dict[Tuple[Any, ...], dict] = {}

    unique_paths = sorted({p for row in all_dirs for p in row})
    logger.info("Unique experiment folders: %d", len(unique_paths))

    for p in unique_paths:
        build_or_get_experiment(exp_cache, p, device, do_compile=(not args.no_compile))

    bp_test_cache: Dict[Tuple[Any, ...], dict] = {}
    bp_train_cache: Dict[Tuple[Any, ...], dict] = {}

    if args.precompute_bp:
        logger.info("Precomputing BP packs (f=0 only; other f cached lazily)...")
        for p in tqdm(unique_paths):
            _ = get_bp_test_pack(
                eval_exp=exp_cache[p],
                device=device,
                schedule_cache=schedule_cache,
                bp_test_cache=bp_test_cache,
                t_final_test_override=args.t_final_test,
                t_denoising_ratio=args.t_denoising_ratio,
                N_test=args.N_test,
                reps=args.reps,
                test_block_i=args.test_block_i,
            )
        if has_train:
            for p in tqdm(unique_paths):
                _ = get_bp_train_pack_for_model1(
                    eval_exp=exp_cache[p],
                    device=device,
                    schedule_cache=schedule_cache,
                    bp_train_cache=bp_train_cache,
                    t_final_test_override=args.t_final_test,
                    t_denoising_ratio=args.t_denoising_ratio,
                    N_test=args.N_test,
                    reps=args.reps,
                    test_block_i=args.test_block_i,
                    print_overlap_once=args.print_overlap,
                )

    pack0 = get_bp_test_pack(
        eval_exp=exp_cache[unique_paths[0]],
        device=device,
        schedule_cache=schedule_cache,
        bp_test_cache=bp_test_cache,
        t_final_test_override=args.t_final_test,
        t_denoising_ratio=args.t_denoising_ratio,
        N_test=args.N_test,
        reps=args.reps,
        test_block_i=args.test_block_i,
    )
    title_time = int(pack0["t_noise"])
    tree_depth = int(pack0["tree_depth"])

    for f in f_list:
        if not (0 <= f <= tree_depth):
            raise ValueError(f"factorized_layers={f} invalid for tree_depth={tree_depth}")

    kl_m1_m2_test_ep = []
    kl_m1_m2_test_ep_se = []
    kl_m1_m2_train_ep = []
    kl_m1_m2_train_ep_se = []

    kl_bp_f_m1_test = {f: [] for f in f_list}
    kl_bp_f_m1_test_se = {f: [] for f in f_list}
    kl_bp_f_m2_test = {f: [] for f in f_list}
    kl_bp_f_m2_test_se = {f: [] for f in f_list}

    kl_bp_f_m1_train = {f: [] for f in f_list}
    kl_bp_f_m1_train_se = {f: [] for f in f_list}
    kl_bp_f_m2_train = {f: [] for f in f_list}
    kl_bp_f_m2_train_se = {f: [] for f in f_list}

    reduced_length_name = int(exp_cache[unique_paths[0]]["params"]["reduced_length"])

    logger.info("Scanning for common checkpoints across %d models...", len(unique_paths))
    common_epochs_set = None
    for p in unique_paths:
        ckpts = list(p.glob("test_model_script_epoch_*.pt"))
        starts = [str(c.stem).split("_epoch_")[1] for c in ckpts]
        epochs_p = {int(s) for s in starts if s.isdigit()}
        common_epochs_set = epochs_p if common_epochs_set is None else common_epochs_set.intersection(epochs_p)

    if not common_epochs_set:
        raise RuntimeError(f"No common epochs found across models: {unique_paths}")

    epochs_to_load = sorted(list(common_epochs_set))
    logger.info(
        "Found %d common epochs. Range: %d .. %d",
        len(epochs_to_load),
        epochs_to_load[0],
        epochs_to_load[-1],
    )

    n_samples = args.N_test * args.reps

    for epoch in epochs_to_load:
        logger.info("################ Processing epoch %d ################", epoch)
        load_epoch_weights_once(unique_paths, exp_cache, epoch, device)

        pred_cache_epoch: Dict[Tuple[Any, ...], np.ndarray] = {}

        kl_m1_m2_test_all = np.zeros((n_samples,), dtype=np.float64)
        kl_m1_m2_train_all = np.zeros((n_samples,), dtype=np.float64)

        bp_f_m1_test_accum = {f: np.zeros((n_samples,), dtype=np.float64) for f in f_list}
        bp_f_m2_test_accum = {f: np.zeros((n_samples,), dtype=np.float64) for f in f_list}
        bp_f_m1_train_accum = {f: np.zeros((n_samples,), dtype=np.float64) for f in f_list}
        bp_f_m2_train_accum = {f: np.zeros((n_samples,), dtype=np.float64) for f in f_list}

        denom = float(len(pairs))

        for (path1, path2) in tqdm(pairs, desc=f"Pairs @ epoch {epoch}", leave=False):
            e1 = exp_cache[Path(path1)]
            e2 = exp_cache[Path(path2)]

            if (e1["vocab_size"] != e2["vocab_size"]) or (e1["seq_len"] != e2["seq_len"]):
                logger.warning("WARNING! Skipping pair due to shape mismatch")
                continue

            bp_test_pack = get_bp_test_pack(
                eval_exp=e1,
                device=device,
                schedule_cache=schedule_cache,
                bp_test_cache=bp_test_cache,
                t_final_test_override=args.t_final_test,
                t_denoising_ratio=args.t_denoising_ratio,
                N_test=args.N_test,
                reps=args.reps,
                test_block_i=args.test_block_i,
            )
            dkey = bp_test_pack["dataset_key"]
            sched = bp_test_pack["schedule"]
            t_noise = bp_test_pack["t_noise"]
            t_final = bp_test_pack["t_final"]

            m1_test = denoise_cached(
                pred_cache_epoch=pred_cache_epoch,
                model_path=Path(path1),
                model=e1["model"],
                x_noised=bp_test_pack["x_test_noised"],
                split="test",
                dataset_key_str=dkey,
                t_noise=t_noise,
                t_final=t_final,
                sched=sched,
                pick_i=None,
                device=device,
            )
            m2_test = denoise_cached(
                pred_cache_epoch=pred_cache_epoch,
                model_path=Path(path2),
                model=e2["model"],
                x_noised=bp_test_pack["x_test_noised"],
                split="test",
                dataset_key_str=dkey,
                t_noise=t_noise,
                t_final=t_final,
                sched=sched,
                pick_i=None,
                device=device,
            )

            kl_m1_m2_test_all += kl_divergence_batch(m1_test, m2_test) / denom

            for f in f_list:
                bp_preds = get_cached_bp_factorized(bp_test_pack, f, backend=args.bp_backend)
                bp_f_m1_test_accum[f] += kl_divergence_batch(bp_preds, m1_test) / denom
                bp_f_m2_test_accum[f] += kl_divergence_batch(bp_preds, m2_test) / denom

            if has_train:
                bp_train_pack = get_bp_train_pack_for_model1(
                    eval_exp=e1,
                    device=device,
                    schedule_cache=schedule_cache,
                    bp_train_cache=bp_train_cache,
                    t_final_test_override=args.t_final_test,
                    t_denoising_ratio=args.t_denoising_ratio,
                    N_test=args.N_test,
                    reps=args.reps,
                    test_block_i=args.test_block_i,
                    print_overlap_once=(args.print_overlap and (not args.precompute_bp)),
                )

                sched_tr = bp_train_pack["schedule"]
                t_noise_tr = bp_train_pack["t_noise"]
                t_final_tr = bp_train_pack["t_final"]
                pick_i_model1 = bp_train_pack["pick_i"]

                m1_train = denoise_cached(
                    pred_cache_epoch=pred_cache_epoch,
                    model_path=Path(path1),
                    model=e1["model"],
                    x_noised=bp_train_pack["x_train_noised"],
                    split="train",
                    dataset_key_str=dkey,
                    t_noise=t_noise_tr,
                    t_final=t_final_tr,
                    sched=sched_tr,
                    pick_i=pick_i_model1,
                    device=device,
                )
                m2_train = denoise_cached(
                    pred_cache_epoch=pred_cache_epoch,
                    model_path=Path(path2),
                    model=e2["model"],
                    x_noised=bp_train_pack["x_train_noised"],
                    split="train",
                    dataset_key_str=dkey,
                    t_noise=t_noise_tr,
                    t_final=t_final_tr,
                    sched=sched_tr,
                    pick_i=pick_i_model1,
                    device=device,
                )

                kl_m1_m2_train_all += kl_divergence_batch(m1_train, m2_train) / denom

                for f in f_list:
                    bp_preds = get_cached_bp_factorized(bp_train_pack, f, backend=args.bp_backend)
                    bp_f_m1_train_accum[f] += kl_divergence_batch(bp_preds, m1_train) / denom
                    bp_f_m2_train_accum[f] += kl_divergence_batch(bp_preds, m2_train) / denom

        m, se = mean_and_se(kl_m1_m2_test_all)
        kl_m1_m2_test_ep.append(m)
        kl_m1_m2_test_ep_se.append(se)

        for f in f_list:
            m, se = mean_and_se(bp_f_m1_test_accum[f])
            kl_bp_f_m1_test[f].append(m)
            kl_bp_f_m1_test_se[f].append(se)

            m, se = mean_and_se(bp_f_m2_test_accum[f])
            kl_bp_f_m2_test[f].append(m)
            kl_bp_f_m2_test_se[f].append(se)

        if has_train:
            m, se = mean_and_se(kl_m1_m2_train_all)
            kl_m1_m2_train_ep.append(m)
            kl_m1_m2_train_ep_se.append(se)

            for f in f_list:
                m, se = mean_and_se(bp_f_m1_train_accum[f])
                kl_bp_f_m1_train[f].append(m)
                kl_bp_f_m1_train_se[f].append(se)

                m, se = mean_and_se(bp_f_m2_train_accum[f])
                kl_bp_f_m2_train[f].append(m)
                kl_bp_f_m2_train_se[f].append(se)

        epochs_plot_so_far = epochs_to_load[: len(kl_m1_m2_test_ep)]

        # legacy: choose f=0 if available else first f
        f0 = 0 if 0 in f_list else f_list[0]

        def get_f0(d_val, d_se):
            return (d_val.get(f0, []), d_se.get(f0, []))

        l_bp_m1_test, l_bp_m1_test_se = get_f0(kl_bp_f_m1_test, kl_bp_f_m1_test_se)
        l_bp_m2_test, l_bp_m2_test_se = get_f0(kl_bp_f_m2_test, kl_bp_f_m2_test_se)

        l_bp_m1_train, l_bp_m1_train_se = ([], [])
        l_bp_m2_train, l_bp_m2_train_se = ([], [])
        if has_train:
            l_bp_m1_train, l_bp_m1_train_se = get_f0(kl_bp_f_m1_train, kl_bp_f_m1_train_se)
            l_bp_m2_train, l_bp_m2_train_se = get_f0(kl_bp_f_m2_train, kl_bp_f_m2_train_se)

        plot_and_save(
            args_obj=args,
            plots_dir=plots_dir,
            epochs_plot=epochs_plot_so_far,
            title_time=title_time,
            t_denoising_ratio=args.t_denoising_ratio,
            reduced_length=reduced_length_name,
            reps=args.reps,
            N_test=args.N_test,
            are_same=are_same,
            has_train=has_train,
            kl_bp_m1_test=l_bp_m1_test,
            kl_bp_m1_test_se=l_bp_m1_test_se,
            kl_bp_m2_test=l_bp_m2_test,
            kl_bp_m2_test_se=l_bp_m2_test_se,
            kl_m1_m2_test=kl_m1_m2_test_ep,
            kl_m1_m2_test_se=kl_m1_m2_test_ep_se,
            kl_bp_m1_train=l_bp_m1_train,
            kl_bp_m1_train_se=l_bp_m1_train_se,
            kl_bp_m2_train=l_bp_m2_train,
            kl_bp_m2_train_se=l_bp_m2_train_se,
            kl_m1_m2_train=kl_m1_m2_train_ep,
            kl_m1_m2_train_se=kl_m1_m2_train_ep_se,
            kl_bp_f_m1_test=kl_bp_f_m1_test,
            kl_bp_f_m1_test_se=kl_bp_f_m1_test_se,
            kl_bp_f_m2_test=kl_bp_f_m2_test,
            kl_bp_f_m2_test_se=kl_bp_f_m2_test_se,
            kl_bp_f_m1_train=kl_bp_f_m1_train,
            kl_bp_f_m1_train_se=kl_bp_f_m1_train_se,
            kl_bp_f_m2_train=kl_bp_f_m2_train,
            kl_bp_f_m2_train_se=kl_bp_f_m2_train_se,
            tag="latest",
            job_id=args.job_id,
        )


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument(
        "--base-path1",
        type=str,
        help="Base model directory with {} placeholder for data_id.",
    )
    p.add_argument("--same-dataset", type=int, default=0)

    p.add_argument("--N-test", dest="N_test", type=int, default=3000)
    p.add_argument("--train-size", dest="train_size", type=int, default=12000)
    p.add_argument("--t-final-test", dest="t_final_test", type=int, default=None)

    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--t-denoising-ratio", type=float, default=0.15)

    p.add_argument("--has-train", type=int, default=1)

    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--test-block-i", type=int, default=15, help="Preferred test block index.")
    p.add_argument(
        "--print-overlap",
        action="store_true",
        help="Print train/test overlap when the BP(TRAIN) pack is first created.",
    )
    p.add_argument(
        "--precompute-bp",
        action="store_true",
        help="Precompute BP caches up front (otherwise lazy).",
    )

    p.add_argument(
        "--keep-epoch-plots",
        action="store_true",
        help="Also save a separate PNG/NPZ snapshot for each epoch.",
    )

    p.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Absolute root directory for models. Bypasses discovery.",
    )
    p.add_argument(
        "--subpath1",
        type=str,
        default=None,
        help="Subpath to M1 model dir (relative to models-root)",
    )
    p.add_argument(
        "--subpath2",
        type=str,
        default=None,
        help="Subpath to M2 model dir (relative to models-root)",
    )
    p.add_argument(
        "--bp-factorized-layers",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated factorized_layers to sweep. 0 means exact.",
    )
    p.add_argument("--job-id", type=str, default=None, help="SLURM job ID to append to output files.")
    p.add_argument(
        "--bp-backend",
        type=str,
        choices=["torch", "numpy"],
        default="torch",
        help="Backend for BP computation: 'torch' (default, GPU) or 'numpy' (CPU).",
    )
    p.add_argument("--output-root", type=str, default=None,
                   help="Root dir for output plots (default: ../plots)")

    args = p.parse_args()
    main(args)
