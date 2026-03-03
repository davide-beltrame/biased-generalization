"""
uturn_overlap_random.py - U-turn on uniform random starting sequences
Referred to as exp3_new in output directory and plotting functions.
Reproduces appendix Fig. 11 (U-turns on random starting sequences).
=====================================================================

Tests denoising performance on uniform random sequences (tokens drawn
uniformly from vocabulary) to probe short-time behavior (regime (iii) in
Sec. 4.2). Compares:
  a. The trained model (learned denoiser).
  b. Empirical score (memorizer: pulls toward training set).
  c. Belief Propagation (BP) with factorized_layers 0-4 (grammar-based denoiser).

Uniform random sequences are completely OOD, so this experiment reveals
whether the trained denoiser and BP coincide on these inputs. The paper
shows they agree only by coincidence: the model learns a simpler rounding
rule rather than the true posterior.

Procedure:
For each epoch and noise level t:
1. Generate N_test uniform random sequences (torch.randint).
2. Add noise via forward diffusion to timestep t.
3. Denoise back to t=0 using Model, Empirical, and BP (k=0..4).
4. Measure reconstruction overlap (fraction of positions matching original).
5. Plot all curves across noise levels.

Outputs:
- Raw overlap arrays (saved in raw/ subdirectory).
- Plot: Model vs Empirical vs BP (k=0..4) overlap with SEM error bars.

Usage (run from scripts/ directory)::

    # U-turn on random starts (App. Fig. 11, n=12k, single seed)
    python uturn_overlap_random.py \\
        --output-root ../plots/ \\
        --paths-gen "<model_path>" \\
        --bp-backend torch \\
        --N-test 100 --reps 100 --grid 5 \\
        --T-min 0.05 --T-max 0.1 --empirical-size 256

    Epoch selection: App. Fig. 11 plots several epochs in different colors
    to show progressive alignment with BP. Omit --which-epoch to scan all
    available checkpoints; or pick a few representative ones (e.g. early,
    NN-divergence minimum, test-loss minimum).
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
import matplotlib.cm as cm

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

# Add project root to path
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
        np.ndarray: Per-sequence overlap (fraction of matching positions)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("counts expects 2D arrays of shape (N, L).")
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
    """Entry point for random-sequence U-turn overlap (Exp 3n / appendix).

    Same protocol as ``uturn_overlap.py`` but with random initial sequences
    instead of training-set seeds.

    Args:
        args (argparse.Namespace): CLI arguments.
    """
    device = setup_device()

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
        
        # Construct BP params tuple: (M, l, q) = (rho, k, vocab_size)
        # NOTE: 'k' from load_data is the tree depth (l in BP), not factorized_layers
        bp_params = (rho, k, vocab_size)
        logger.info("BP params: rho shape=%s, k=%d, vocab_size=%d", rho.shape if hasattr(rho, 'shape') else 'scalar', k, vocab_size)
        
        # DIAGNOSTICS: Check rho structure
        # If rho is uniform, the tree is effectively random and BP won't enforce structure
        rho_np = rho.cpu().numpy() if hasattr(rho, 'cpu') else np.array(rho)
        logger.info("=== RHO DIAGNOSTICS ===")
        logger.info("  rho_mean=%.6f, rho_min=%.6f, rho_max=%.6f", 
                    rho_np.mean(), rho_np.min(), rho_np.max())
        logger.info("  rho_std=%.6f (if close to0, grammar is uniform/uninformative)", rho_np.std())
        # Check if rho is nearly uniform (all values similar)
        if rho_np.std() < 1e-6:
            logger.warning("Rho is uniform, thus BP will not enforce any structure.")
        else:
            logger.info("Rho is structured, thus BP should enforce grammar.")

        # Training set selection logic (for empirical score x0s)
        pick_i = int(params.get("pick_i_for_training", 0) or 0)
        li = pick_i*reduced_length
        ri = li + reduced_length
        train_seqs = sequences[li:ri]
        
        logger.info("Training set size: %d (for Empirical Score)", len(train_seqs))
        logger.info("Evaluation: UNIFORM RANDOM NOISE")

        # One-hot mapping for training sequences
        train_seqs_one_hot = torch.nn.functional.one_hot(
            train_seqs.long(), num_classes=vocab_size
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

        # Saving directory
        _plots_base = Path(args.output_root) if args.output_root else PROJECT_ROOT / "plots"
        plots_dir = (
            _plots_base
            / "exp3_new"
            / f"N_train_{params['reduced_length']}"
            / f"N_test_{args.N_test}"
            / f"pick_i_{pick_i}"
            / f"seed_{params['seed']}"
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

        # Time grid construction
        T_min = float(args.T_min)
        T_max = float(args.T_max)
        grid = int(args.grid)
        t_range = np.linspace(T_min, T_max, grid, dtype=float)
        logger.info("Noising steps grid (t/T_final): %s", t_range.tolist())

        # Subsample training sequences for empirical score
        empirical_size = min(args.empirical_size, len(train_seqs))
        logger.info("Subsampling training set from %d to %d for empirical score", len(train_seqs), empirical_size)
        subsample_indices = np.random.RandomState(args.seed).choice(
            len(train_seqs), empirical_size, replace=False
        )
        train_seqs_subsampled = train_seqs[subsample_indices]
        train_seqs_subsampled_one_hot = train_seqs_one_hot[subsample_indices]

        # Epoch loop
        sub_plots_dir = plots_dir
        reps = int(args.reps)
        
        # BP factorized layers to evaluate
        bp_fl_list = list(range(5))
        
        for i_ep, epoch in enumerate(epochs):
            # Containers for overlap values
            avg_model_overlap_list = []
            avg_empirical_overlap_list = []
            avg_model_overlap_sem_list = []
            avg_empirical_overlap_sem_list = []
            
            # BP containers: one list per factorized_layers value
            avg_bp_overlap_lists = {fl: [] for fl in bp_fl_list}
            avg_bp_overlap_sem_lists = {fl: [] for fl in bp_fl_list}

            logger.info("Epoch %d", epoch)
            ckpt = model_dir / f"test_model_script_epoch_{epoch}.pt"
            if (not ckpt.exists()) or (not load_checkpoint(model, str(ckpt), device, training=False)):
                logger.warning("Checkpoint %d failed", epoch)
                continue
            model.eval()

            # Loop over denoising time (U-turn)
            for i_t, t_denoising_ratio in enumerate(t_range):
                t_denoising = torch.tensor(int(t_denoising_ratio * t_final), device=device, dtype=torch.long)
                logger.info("T denoising ratio = %.4f (step = %d)", t_denoising_ratio, t_denoising.item())

                # Precompute paired noised starts and clean sequences for each rep
                paired_noised_starts = []
                paired_clean_seqs = []
                
                for iter_gen in range(reps):
                    # Seed is deterministic per (epoch, t, rep) for reproducibility
                    seed_rep = _paired_seed(args.seed, i_ep, i_t, iter_gen)
                    _seed_all(seed_rep)
                    
                    # Generate uniform random sequences
                    eval_seqs = torch.randint(0, vocab_size, (args.N_test, seq_len), device=device)
                    eval_seqs_one_hot = torch.nn.functional.one_hot(
                        eval_seqs.long(), num_classes=vocab_size
                    ).float()
                    
                    # Noise them forward (add diffusion noise on top of uniform random)
                    x_noised = forward_process(
                        eval_seqs_one_hot, t_denoising, alpha_bars=alpha_bars
                    )
                    
                    paired_noised_starts.append(x_noised)
                    paired_clean_seqs.append(eval_seqs_one_hot)

                # Model denoising: learned denoiser performance
                model_overlaps = []
                for iter_gen in range(reps):
                    x_noised = paired_noised_starts[iter_gen].detach().clone()
                    eval_seqs_one_hot = paired_clean_seqs[iter_gen]
                    logger.info(f"########## MODEL REPETITION NUMBER: {iter_gen} ##########")

                    # Denoise using the trained model
                    x0_hat = generate_sequences(
                        args.N_test, model,
                        t_denoising.item(),
                        alpha_bars, alphas,
                        seq_len, in_channels, device,
                        clamp=False, temperature=None,
                        batch_size=args.batch_size if args.batch_size else args.N_test,
                        bp_params=None,
                        start_seqs=x_noised,
                        fix_noise=False
                    ).cpu().numpy()
                    
                    # Check overlap with original clean sequences
                    ov = counts(x0_hat, eval_seqs_one_hot.argmax(2).cpu().numpy())
                    model_overlaps.append(ov)

                model_overlaps = np.concatenate(model_overlaps, dtype=float)

                # Empirical score denoising: ground truth from training data
                empirical_overlaps = []
                for iter_gen in range(reps):
                    x_noised = paired_noised_starts[iter_gen].detach().clone()
                    eval_seqs_one_hot = paired_clean_seqs[iter_gen]
                    logger.info(f"########## EMPIRICAL REPETITION NUMBER: {iter_gen} ##########")

                    # Denoise via empirical score
                    x0_emp = generate_sequences(
                        args.N_test, model,
                        t_denoising.item(),
                        alpha_bars, alphas,
                        seq_len, in_channels, device,
                        clamp=False, temperature=None,
                        batch_size=args.batch_size if args.batch_size else args.N_test,
                        bp_params=None,
                        start_seqs=x_noised,
                        empirical_score=True,
                        x0s=train_seqs_subsampled_one_hot,
                        fix_noise=False
                    ).cpu().numpy()
                    
                    # Check overlap with original clean sequences
                    ov = counts(x0_emp, eval_seqs_one_hot.argmax(2).cpu().numpy())
                    empirical_overlaps.append(ov)

                empirical_overlaps = np.concatenate(empirical_overlaps, dtype=float)

                # BP denoising: theoretical optimal with factorized_layers 0-4
                bp_overlaps = {fl: [] for fl in bp_fl_list}
                for fl in bp_fl_list:
                    logger.info(f"########## BP (k={fl}) ##########")
                    for iter_gen in range(reps):
                        x_noised = paired_noised_starts[iter_gen].detach().clone()
                        eval_seqs_one_hot = paired_clean_seqs[iter_gen]
                        logger.info(f"BP (k={fl}) REPETITION NUMBER: {iter_gen}")

                        # Denoise via BP with factorized_layers=fl
                        x0_bp = generate_sequences(
                            args.N_test, model,
                            t_denoising.item(),
                            alpha_bars.cpu().numpy(), alphas.cpu().numpy(),
                            seq_len, in_channels, device,
                            clamp=False, temperature=None,
                            batch_size=args.batch_size if args.batch_size else args.N_test,
                            bp_params=bp_params,
                            start_seqs=x_noised,
                            factorized_layers=fl,
                            fix_noise=False,
                            bp_backend=args.bp_backend,
                        ).cpu().numpy()
                        
                        # Check overlap with original clean sequences
                        ov = counts(x0_bp, eval_seqs_one_hot.argmax(2).cpu().numpy())
                        bp_overlaps[fl].append(ov)
                    
                    bp_overlaps[fl] = np.concatenate(bp_overlaps[fl], dtype=float)

                # Save raw overlap arrays for this epoch and t step
                tstep = int(t_denoising.item())
                tag = f"epoch_{epoch}_tstep_{tstep}_Ntest_{args.N_test}_seed_{args.seed}_reps_{reps}"
                np.save(arrays_dir / f"model_overlap_{tag}.npy", model_overlaps)
                np.save(arrays_dir / f"empirical_overlap_{tag}.npy", empirical_overlaps)
                for fl in bp_fl_list:
                    np.save(arrays_dir / f"bp_k{fl}_overlap_{tag}.npy", bp_overlaps[fl])

                n_per_rep = int(args.N_test)
                # Reshape to (reps, N_test): each row is one repetition
                M = model_overlaps.reshape(reps, n_per_rep)
                E = empirical_overlaps.reshape(reps, n_per_rep)

                # Mean across repetitions for each sample
                # Shape: (N_test,) one mean overlap value per original sequence
                mean_M = np.nanmean(M, axis=0)
                mean_E = np.nanmean(E, axis=0)

                # Average overlap across all N_test samples (with SEM for error bars)
                avg_model_overlap_list.append(float(np.nanmean(mean_M)))
                avg_empirical_overlap_list.append(float(np.nanmean(mean_E)))
                avg_model_overlap_sem_list.append(_sem_over_valid(mean_M))
                avg_empirical_overlap_sem_list.append(_sem_over_valid(mean_E))
                
                # BP averages
                for fl in bp_fl_list:
                    BP = bp_overlaps[fl].reshape(reps, n_per_rep)
                    mean_BP = np.nanmean(BP, axis=0)
                    avg_bp_overlap_lists[fl].append(float(np.nanmean(mean_BP)))
                    avg_bp_overlap_sem_lists[fl].append(_sem_over_valid(mean_BP))

                # Plots (average overlaps with SEM error bars)
                times = [t for t in t_range[:(i_t + 1)]]

                model_ov = np.array(avg_model_overlap_list)
                empirical_ov = np.array(avg_empirical_overlap_list)
                model_ov_sem = np.array(avg_model_overlap_sem_list)
                empirical_ov_sem = np.array(avg_empirical_overlap_sem_list)

                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Model
                ax.errorbar(
                    times, model_ov, yerr=model_ov_sem,
                    fmt='-o', markersize=4, linewidth=1.8, capsize=3, 
                    color='tab:red', label='Model'
                )
                ax.fill_between(
                    times,
                    model_ov - model_ov_sem,
                    model_ov + model_ov_sem,
                    alpha=0.12, color='tab:red'
                )
                
                # Empirical
                ax.errorbar(
                    times, empirical_ov, yerr=empirical_ov_sem,
                    fmt='-s', markersize=4, linewidth=1.8, capsize=3, 
                    color='tab:orange', label='Empirical'
                )
                ax.fill_between(
                    times,
                    empirical_ov - empirical_ov_sem,
                    empirical_ov + empirical_ov_sem,
                    alpha=0.12, color='tab:orange'
                )
                
                # BP curves
                bp_colors = cm.Blues(np.linspace(0.4, 0.9, len(bp_fl_list)))
                for i, fl in enumerate(bp_fl_list):
                    bp_ov = np.array(avg_bp_overlap_lists[fl])
                    bp_ov_sem = np.array(avg_bp_overlap_sem_lists[fl])
                    ax.errorbar(
                        times, bp_ov, yerr=bp_ov_sem,
                        fmt='-^', markersize=3, linewidth=1.5, capsize=2,
                        color=bp_colors[i], label=f'BP (k={fl})'
                    )
                    ax.fill_between(
                        times,
                        bp_ov - bp_ov_sem,
                        bp_ov + bp_ov_sem,
                        alpha=0.08, color=bp_colors[i]
                    )
                
                ax.set_xlabel('t / T_final')
                ax.set_ylabel('Reconstruction overlap (fraction correct)')
                ax.set_title(f'Reconstruction of Uniform Noise (epoch {epoch})')
                ax.grid(True, linestyle=':', linewidth=0.8)
                for spine in ('top', 'right'):
                    ax.spines[spine].set_visible(False)
                ax.legend(loc='best', fontsize=9, ncol=2)
                fig.tight_layout()
                path = sub_plots_dir / f"model_vs_empirical_vs_bp_epoch_{epoch}_Ntest_{args.N_test}_reps_{args.reps}.png"
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)

    logger.info("Done.")


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--paths-gen', type=str,
                   default="results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_20000_5000_cross_entropy_loss/8_512",
                   help='Comma-separated list of model generation directories')
    p.add_argument("--N-test", type=int, default=100)
    p.add_argument("--which-epoch", type=int, default=None,
                   help="Single epoch to evaluate. Paper App. Fig. 11 shows "
                        "several training epochs in different colors to "
                        "illustrate progressive alignment with BP.")
    p.add_argument("--start-epoch", type=int, default=None,
                   help="Discard checkpoints before this epoch.")
    p.add_argument("--max-epoch", type=int, default=None,
                   help="Discard checkpoints after this epoch.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--T-min", type=float, default=0.05, help="Minimum t/T_final ratio")
    p.add_argument("--T-max", type=float, default=0.1, help="Maximum t/T_final ratio")
    p.add_argument("--empirical-size", type=int, default=256, help="Subsample size for empirical score training data")
    p.add_argument("--model-type", type=str, default='transformer', help="Model type to use.")
    p.add_argument("--bp-backend", type=str, choices=['torch', 'numpy'], default='torch',
                   help="Backend for belief propagation: 'torch' (batched GPU) or 'numpy' (serial CPU)")
    p.add_argument("--output-root", type=str, default=None,
                   help="Root dir for output plots (default: PROJECT_ROOT/plots)")
    args = p.parse_args()
    main(args)
