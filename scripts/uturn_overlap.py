"""
uturn_overlap.py - U-turn experiment for measuring biased generalization
Referred to as exp3 in output directory and plotting functions.
Reproduces Fig. 5 and appendix Figs. 9(right), 10(right).
================================================================

Measures model bias by comparing reconstruction accuracy after a "U-turn":
forward diffusion (noise) followed by backward diffusion (denoise) from a
specific timestep t. If a model is biased toward the training distribution, it
will reconstruct TRAIN sequences more accurately than TEST sequences.

The ratio model_overlap / BP_overlap normalizes for the non-uniform data
distribution (different starting points have different expected overlaps even
under the exact BP denoiser) and isolates deviations of the trained model
from oracle behavior.

Procedure:
For each epoch and noise level t:
1. Take N_test sequences from training set and N_test from test set.
2. Add noise via forward diffusion to timestep t.
3. Denoise back to t=0 using:
   a. The trained model (learned denoiser).
   b. Belief Propagation (BP) - theoretically optimal given the grammar.
4. Measure reconstruction overlap (fraction of positions matching original).
5. Compute ratio = model_overlap / BP_overlap for both train and test.
6. Compare train ratio vs test ratio across noise levels.

Outputs:
- Raw overlap arrays (saved in raw/ subdirectory).
- Plot: model/BP overlap ratio vs noise level for train and test.

Interpretation:
- Ratio = 1: model matches BP (oracle performance).
- Ratio < 1: model reconstructs worse than BP (expected).
- Train ratio > Test ratio: model has bias toward training distribution.
- At NN-divergence minimum, train and test curves are indistinguishable;
  at the test-loss minimum, a clear separation emerges.

Usage (run from scripts/ directory)::

    # U-turn overlap (Fig. 5, n=12k, 3 models: 2 same-dataset + 1 disjoint)
    python uturn_overlap.py \\
        --output-root ../plots/ \\
        --paths-gen "<path_seed0>,<path_seed1>,<path_disjoint_seed2>" \\
        --bp-backend torch \\
        --N-test 1000 --T-max 0.12

    Epoch selection: the paper (Fig. 5) compares two checkpoints for
    each model, the NN-divergence minimum and the test-loss minimum.
    Run nn_divergence.py first and pick the two relevant epochs via
    --which-epoch (one run per epoch).
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

# Plot style
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.frameon": False,
    "grid.alpha": 0.3,
})

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.testing import generate_sequences
from modules.diffusion import forward_process
from utils import (
    load_params,
    load_data,
    setup_device,
    create_model,
    load_checkpoint,
    find_available_epochs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def counts(a, b):
    """
    Overlap per sequence: fraction of positions that match.
    
    Args:
        a: (N, L) integer array of predicted sequences
        b: (N, L) integer array of ground truth sequences
    
    Returns:
        np.ndarray: per-sequence overlap (fraction of matching positions)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Counts expects 2D arrays of shape (N, L).")
    N, L = a.shape
    if b.shape[1] != L:
        raise ValueError(f"Mismatched sequence length: a has L={L}, b has L={b.shape[1]}.")

    bN = b[:N]
    per_seq_overlap = (a == bN).mean(axis=1)
    return per_seq_overlap

def _paired_seed(base_seed, epoch_idx, t_idx, rep_idx):
    """Deterministic per-(epoch,t,rep) seed so we can reproduce the same noised starts."""
    if base_seed is None:
        base_seed = 0
    return int(base_seed + 1_000_003 * epoch_idx + 10_007 * t_idx + rep_idx)

def _seed_all(seed: int):
    """Seed torch and CUDA (if available) deterministically."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _sem_over_valid(x):
    """Return SEM over finite values; NaN if <2 valid."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(n))

def main(args):
    """Entry point for U-turn overlap experiment (Exp 3 / Fig 5).

    For each epoch and denoising ratio, generates sequences via the backward
    process (from noise or U-turn) and measures NN overlap with the training
    set.

    Args:
        args (argparse.Namespace): CLI arguments.
    """
    # Set GPU if available
    device = setup_device()

    # Parse comma‐list of directories
    paths = [Path(p.strip()) for p in args.paths_gen.split(",") if p.strip()]
    if not paths:
        logger.error("No --paths_gen provided")
        return

    # Loop over model paths
    for model_dir in paths:
        logger.info("Processing %s", model_dir)

        # Load params & data
        params = load_params(model_dir / "full_params.json")
        reduced_length = params['reduced_length']
        vocab_size, sequences, seq_len, k, rho, _ = load_data(
            params['data_path'], device
        )

        # Training/test set selection logic
        pick_i = int(params.get("pick_i_for_training", 0) or 0)
        li = pick_i*reduced_length
        ri = li + reduced_length
        train_seqs = sequences[li:ri]

        test_seqs = sequences[ri:ri+args.N_test]

        # One-hot mapping
        train_seqs_one_hot = torch.nn.functional.one_hot(
            train_seqs.long(), num_classes=vocab_size
        ).float().to(device)
        test_seqs_one_hot = torch.nn.functional.one_hot(
            test_seqs.long(), num_classes=vocab_size
        ).float().to(device)

        # Model setup
        model, in_channels, _ = create_model(params, vocab_size, seq_len, device)
        model = torch.compile(model)
        logger.info("Model instantiated")

        # Diffusion schedule
        t_final = params['t_final']
        t_test = t_final
        alpha_bars = alpha_bars_schedule(
            t_test, device=device, s=params['s'],
            schedule='linear'
        )
        alphas, _ = get_alpha_beta_from_alpha_bar(alpha_bars)

        # Save directories
        _plots_base = Path(args.output_root) if args.output_root else PROJECT_ROOT / "plots"
        plots_dir = (
            _plots_base
            / "exp3"
            / f"N_train_{params['reduced_length']}"
            / f"N_test_{args.N_test}"
            / f"pick_i_{pick_i}"
            / f"seed_{params['seed']}"
            / f"backend_{args.bp_backend}"
        )
        plots_dir.mkdir(parents=True, exist_ok=True)
        arrays_dir = plots_dir / "raw"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        # Epochs
        epochs = ([args.which_epoch] if args.which_epoch
                  else find_available_epochs(model_dir))
        epochs = sorted(list(set(epochs)))
        if args.max_epoch is not None:
            epochs = [e for e in epochs if e <= args.max_epoch]
        if args.start_epoch is not None:
            epochs = [e for e in epochs if e >= args.start_epoch]
        if not epochs:
            logger.warning("No epochs found in %s", model_dir)
            continue
        logger.info("Found epochs: %s", epochs)

        # Select time steps for evaluation
        T_max = float(args.T_max)
        grid = int(args.grid)
        base_pts = np.array([0.01, 0.03], dtype=float)
        if grid <= len(base_pts):
            t_range = base_pts[:grid]
        else:
            extra = np.linspace(0.05, T_max, grid - len(base_pts), dtype=float) if T_max > 0.05 else np.array([], dtype=float)
            t_range = np.concatenate([base_pts, extra], dtype=float)
        logger.info("Noising steps grid: %s", t_range.tolist())

        # Epoch loop
        sub_plots_dir = plots_dir
        reps = int(args.reps)
        for i_ep, epoch in enumerate(epochs):
            # Containers for ratios between models and BP on train and test data
            avg_ratio_md_bp_train_list = []
            avg_ratio_md_bp_test_list = []
            avg_ratio_md_bp_train_sem_list = []
            avg_ratio_md_bp_test_sem_list = []
            
            # Containers for direct model overlap values (for bias signal plot)
            avg_model_train_overlap_list = []
            avg_model_test_overlap_list = []
            avg_model_train_overlap_sem_list = []
            avg_model_test_overlap_sem_list = []

            logger.info("Epoch %d", epoch)
            # Load checkpoint
            ckpt = model_dir / f"test_model_script_epoch_{epoch}.pt"
            if (not ckpt.exists()) or (not load_checkpoint(model, str(ckpt), device, training=False)):
                logger.warning("Checkpoint %d failed", epoch)
                continue
            model.eval()

            # Loop over denoising time (U-turn)
            for i_t, t_denoising_ratio in enumerate(t_range):
                t_denoising = torch.tensor(int(t_denoising_ratio * t_final), device=device, dtype=torch.long)
                logger.info("T denoising = %d", t_denoising.item())

                # Precompute paired noised starts once per (epoch, t)
                paired_train_starts = []
                paired_test_starts = []
                # Loop over repetitions
                for iter_gen in range(reps):
                    # Seed is deterministic per (epoch, t, rep) for reproducibility.
                    # NOTE: Same seed across epochs means noise pattern is consistent, allowing fair comparison of model performance across training.
                    seed_rep = _paired_seed(args.seed, i_ep, i_t, iter_gen)
                    _seed_all(seed_rep)

                    # Same noised starts for both BP and model
                    x_train_noised = forward_process(
                        train_seqs_one_hot[:args.N_test], t_denoising, alpha_bars=alpha_bars
                    )
                    x_test_noised = forward_process(
                        test_seqs_one_hot[:args.N_test], t_denoising, alpha_bars=alpha_bars
                    )
                    paired_train_starts.append(x_train_noised)
                    paired_test_starts.append(x_test_noised)

                # BP baseline: theoretically optimal denoiser given the grammar
                # We run BP from the same noised starts as the model for fair comparison
                bp_train_overlaps = []
                bp_test_overlaps = []
                for iter_gen in range(reps):
                    # Retrieve noisy sequences for specific rep
                    x_train_noised = paired_train_starts[iter_gen].detach().clone()
                    x_test_noised = paired_test_starts[iter_gen].detach().clone()
                    logger.info(f"########## BP-Empirical REPETITION NUMBER: {iter_gen} ##########")
                    
                    # Train-start BP
                    x0_bp = generate_sequences(
                        args.N_test, model,
                        t_denoising.item(),
                        alpha_bars.cpu().numpy(), alphas.cpu().numpy(),
                        seq_len, in_channels, device,
                        clamp=False, temperature=None,
                        batch_size=args.batch_size if args.batch_size else args.N_test,
                        bp_params=(rho, k, vocab_size),
                        start_seqs=x_train_noised, # we fix the starting sequence to x_train_noised
                        fix_noise=False,  # fresh noise: unbiased E[overlap] over reps
                        bp_backend=args.bp_backend,
                    ).cpu().numpy()
                    # Check overlap with initial clean training sequences
                    ov_train = counts(x0_bp, train_seqs_one_hot.argmax(2).cpu().numpy())
                    bp_train_overlaps.append(ov_train)

                    # Test-start BP
                    x0_bp = generate_sequences(
                        args.N_test, model,
                        t_denoising.item(),
                        alpha_bars.cpu().numpy(), alphas.cpu().numpy(),
                        seq_len, in_channels, device,
                        clamp=False, temperature=None,
                        batch_size=args.batch_size if args.batch_size else args.N_test,
                        bp_params=(rho, k, vocab_size),
                        start_seqs=x_test_noised, # we fix the starting sequence to x_test_noised
                        fix_noise=False,  # fresh noise: unbiased E[overlap] over reps
                        bp_backend=args.bp_backend,
                    ).cpu().numpy()
                    # Check overlap with initial clean test sequences
                    ov_test = counts(x0_bp, test_seqs_one_hot.argmax(2).cpu().numpy())
                    bp_test_overlaps.append(ov_test)

                # Concatenate across repetitions
                bp_train_overlaps = np.concatenate(bp_train_overlaps, dtype=float)
                bp_test_overlaps = np.concatenate(bp_test_overlaps, dtype=float)

                # Model denoising: learned denoiser performance
                # Uses the same noised starts as BP for direct comparison
                model_train_overlaps = []
                model_test_overlaps = []
                for iter_gen in range(reps):
                    x_train_noised = paired_train_starts[iter_gen].detach().clone()
                    x_test_noised = paired_test_starts[iter_gen].detach().clone()
                    logger.info(f"########## MODEL REPETITION NUMBER: {iter_gen} ##########")

                    # Train-start model
                    x0_hat = generate_sequences(
                        args.N_test, model,
                        t_denoising.item(),
                        alpha_bars, alphas,
                        seq_len, in_channels, device,
                        clamp=False, temperature=None,
                        batch_size=args.batch_size if args.batch_size else args.N_test,
                        bp_params=None,
                        start_seqs=x_train_noised,
                        fix_noise=False  # fresh noise: unbiased E[overlap] over reps
                    ).cpu().numpy()
                    # Check overlap with initial clean training sequences
                    ov_train = counts(x0_hat, train_seqs_one_hot.argmax(2).cpu().numpy())
                    model_train_overlaps.append(ov_train)

                    # Test-start model
                    x0_hat = generate_sequences(
                        args.N_test, model,
                        t_denoising.item(),
                        alpha_bars, alphas,
                        seq_len, in_channels, device,
                        clamp=False, temperature=None,
                        batch_size=args.batch_size if args.batch_size else args.N_test,
                        bp_params=None,
                        start_seqs=x_test_noised,
                        fix_noise=False  # fresh noise: unbiased E[overlap] over reps
                    ).cpu().numpy()
                    # Check overlap with initial clean test sequences
                    ov_test = counts(x0_hat, test_seqs_one_hot.argmax(2).cpu().numpy())
                    model_test_overlaps.append(ov_test)

                # Concatenate across repetitions
                model_train_overlaps = np.concatenate(model_train_overlaps, dtype=float)
                model_test_overlaps = np.concatenate(model_test_overlaps, dtype=float)
                
                # Save raw overlap arrays for this epoch and t step
                tstep = int(t_denoising.item())
                tag = f"epoch_{epoch}_tstep_{tstep}_Ntest_{args.N_test}_seed_{args.seed}_reps_{reps}"
                # Train
                np.save(arrays_dir / f"train_model_overlap_{tag}.npy", model_train_overlaps)
                np.save(arrays_dir / f"train_bp_overlap_{tag}.npy", bp_train_overlaps)
                # Test
                np.save(arrays_dir / f"test_model_overlap_{tag}.npy", model_test_overlaps)
                np.save(arrays_dir / f"test_bp_overlap_{tag}.npy", bp_test_overlaps)

                n_per_rep = int(args.N_test)
                # Reshape to (reps, N_test): each row is one repetition
                M_tr = model_train_overlaps.reshape(reps, n_per_rep)
                B_tr = bp_train_overlaps.reshape(reps,  n_per_rep)
                M_te = model_test_overlaps.reshape(reps, n_per_rep)
                B_te = bp_test_overlaps.reshape(reps,  n_per_rep)

                # Mean across repetitions for each sample (reduces noise from stochastic denoising)
                # Shape: (N_test,) - one mean overlap value per original sequence
                mean_M_tr = np.nanmean(M_tr, axis=0)
                mean_B_tr = np.nanmean(B_tr, axis=0)
                mean_M_te = np.nanmean(M_te, axis=0)
                mean_B_te = np.nanmean(B_te, axis=0)

                # Ratio of means per sample: model_overlap / BP_overlap
                # Values < 1 mean model is worse than BP as expected
                # Uses safe division to handle BP=0 edge cases
                ratio_train_rep = np.divide(
                    mean_M_tr, mean_B_tr,
                    out=np.full_like(mean_M_tr, np.nan, dtype=float),
                    where=(mean_B_tr != 0.0)
                )
                ratio_test_rep = np.divide(
                    mean_M_te, mean_B_te,
                    out=np.full_like(mean_M_te, np.nan, dtype=float),
                    where=(mean_B_te != 0.0)
                )

                # Average ratio across all N_test samples (with SEM for error bars)
                avg_ratio_md_bp_train_list.append(float(np.nanmean(ratio_train_rep)))
                avg_ratio_md_bp_test_list.append(float(np.nanmean(ratio_test_rep)))
                avg_ratio_md_bp_train_sem_list.append(_sem_over_valid(ratio_train_rep))
                avg_ratio_md_bp_test_sem_list.append(_sem_over_valid(ratio_test_rep))
                
                # Also track direct model overlap values for bias signal plot
                # This shows: does the model reconstruct train better than test?
                avg_model_train_overlap_list.append(float(np.nanmean(mean_M_tr)))
                avg_model_test_overlap_list.append(float(np.nanmean(mean_M_te)))
                avg_model_train_overlap_sem_list.append(_sem_over_valid(mean_M_tr))
                avg_model_test_overlap_sem_list.append(_sem_over_valid(mean_M_te))

                # Plots (average ratios with SEM error bars)
                times = [t for t in t_range[:(i_t + 1)]]

                ratio_md_bp_train = np.array(avg_ratio_md_bp_train_list)
                ratio_md_bp_test = np.array(avg_ratio_md_bp_test_list)
                ratio_md_bp_train_sem = np.array(avg_ratio_md_bp_train_sem_list)
                ratio_md_bp_test_sem = np.array(avg_ratio_md_bp_test_sem_list)
                
                model_train_ov = np.array(avg_model_train_overlap_list)
                model_test_ov = np.array(avg_model_test_overlap_list)
                model_train_ov_sem = np.array(avg_model_train_overlap_sem_list)
                model_test_ov_sem = np.array(avg_model_test_overlap_sem_list)

                # Plot 1: Average ratio (model / BP) with SEM across N_test
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.errorbar(
                    times, ratio_md_bp_train, yerr=ratio_md_bp_train_sem,
                    fmt='-o', markersize=4, linewidth=1.8, capsize=3, label='Train avg(model/BP)'
                )
                ax.fill_between(
                    times,
                    ratio_md_bp_train - ratio_md_bp_train_sem,
                    ratio_md_bp_train + ratio_md_bp_train_sem,
                    alpha=0.12
                )
                ax.errorbar(
                    times, ratio_md_bp_test, yerr=ratio_md_bp_test_sem,
                    fmt='-s', markersize=4, linewidth=1.8, capsize=3, label='Test avg(model/BP)'
                )
                ax.fill_between(
                    times,
                    ratio_md_bp_test - ratio_md_bp_test_sem,
                    ratio_md_bp_test + ratio_md_bp_test_sem,
                    alpha=0.12
                )
                ax.axhline(1.0, linestyle='--', linewidth=1.2)
                ax.set_xlabel('t / T_final')
                ax.set_ylabel('Average overlap ratio (model / BP)')
                ax.set_title(f'Average(model/BP) vs denoising time  (epoch {epoch})')
                ax.grid(True, linestyle=':', linewidth=0.8)
                for spine in ('top', 'right'):
                    ax.spines[spine].set_visible(False)
                ax.legend(loc='best')
                fig.tight_layout()
                path = sub_plots_dir / f"nn_perc_all_epoch_{epoch}_Ntest_{args.N_test}_reps_{args.reps}.png"
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)

                # Plot 2: “Overfitting” diagnostics (use SEM propagation)
                diff_ratio = np.abs(ratio_md_bp_train - ratio_md_bp_test)
                diff_sem = np.sqrt(np.nan_to_num(ratio_md_bp_train_sem, nan=0.0)**2 +
                                   np.nan_to_num(ratio_md_bp_test_sem,  nan=0.0)**2)
                diff_bias = 0.5 * (np.abs(1.0 - ratio_md_bp_train) + np.abs(1.0 - ratio_md_bp_test))
                bias_sem = 0.5 * diff_sem

                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.plot(times, diff_ratio, '-o', linewidth=1.8, markersize=4, label='|train - test|')
                ax.fill_between(times, diff_ratio - diff_sem, diff_ratio + diff_sem, alpha=0.12)
                ax.plot(times, diff_bias, '-s', linewidth=1.8, markersize=4, label='mean(|1 - ratio|)')
                ax.fill_between(times, diff_bias - bias_sem, diff_bias + bias_sem, alpha=0.12)
                ax.set_xlabel('t / T_final')
                ax.set_ylabel('Distance')
                ax.set_title(f'Train–Test gap and bias vs time  (epoch {epoch})')
                ax.grid(True, linestyle=':', linewidth=0.8)
                for spine in ('top', 'right'):
                    ax.spines[spine].set_visible(False)
                ax.legend(loc='best')
                fig.tight_layout()
                path = sub_plots_dir / f"nn_overfitting_all_epoch_{epoch}_Ntest_{args.N_test}_reps_{args.reps}.png"
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)

                # Plot 3: Direct bias signal - Model reconstruction overlap (train vs test)
                # This is the clearest bias indicator: if model_train > model_test,
                # the model reconstructs training sequences better (biased)
                fig, ax = plt.subplots(figsize=(6.5, 4.2))
                ax.errorbar(
                    times, model_train_ov, yerr=model_train_ov_sem,
                    fmt='-o', markersize=4, linewidth=1.8, capsize=3, label='Train overlap'
                )
                ax.fill_between(
                    times,
                    model_train_ov - model_train_ov_sem,
                    model_train_ov + model_train_ov_sem,
                    alpha=0.12
                )
                ax.errorbar(
                    times, model_test_ov, yerr=model_test_ov_sem,
                    fmt='-s', markersize=4, linewidth=1.8, capsize=3, label='Test overlap'
                )
                ax.fill_between(
                    times,
                    model_test_ov - model_test_ov_sem,
                    model_test_ov + model_test_ov_sem,
                    alpha=0.12
                )
                ax.set_xlabel('t / T_final')
                ax.set_ylabel('Reconstruction overlap (fraction correct)')
                ax.set_title(f'Model reconstruction: Train vs Test  (epoch {epoch})')
                ax.grid(True, linestyle=':', linewidth=0.8)
                for spine in ('top', 'right'):
                    ax.spines[spine].set_visible(False)
                ax.legend(loc='best')
                fig.tight_layout()
                path = sub_plots_dir / f"bias_signal_epoch_{epoch}_Ntest_{args.N_test}_reps_{args.reps}.png"
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)
    logger.info("Done.")


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--paths-gen', type=str,
                   default="results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_20000_5000_cross_entropy_loss/8_512",
                   help='Comma-separated list of model generation directories')
    p.add_argument("--N-test", type=int, default=1000)
    p.add_argument("--which-epoch", type=int, default=None,
                   help="Single epoch to evaluate. Paper Fig. 5 compares two "
                        "checkpoints: the NN-divergence minimum and the "
                        "test-loss minimum. Run nn_divergence.py first to "
                        "identify both.")
    p.add_argument("--start-epoch", type=int, default=None,
                   help="Discard checkpoints before this epoch.")
    p.add_argument("--max-epoch", type=int, default=None,
                   help="Discard checkpoints after this epoch.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--T-max", type=float, default=0.12)
    p.add_argument("--model-type", type=str, default='transformer', help="Model type to use.")
    p.add_argument("--bp-backend", type=str, choices=['torch', 'numpy'], default='torch',
                   help="Backend for belief propagation: 'torch' (batched GPU) or 'numpy' (serial CPU)")
    p.add_argument("--output-root", type=str, default=None,
                   help="Root dir for output plots (default: PROJECT_ROOT/plots)")
    args = p.parse_args()
    main(args)
