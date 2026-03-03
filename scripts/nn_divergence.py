"""
nn_divergence.py - Nearest-neighbor divergence for biased generalization
Referred to as exp2 in output directory and plotting functions.
Reproduces Fig. 4(a) and appendix Figs. 9(left) and 10(left).
================================================================

Measures model bias by comparing the nearest-neighbor (NN) overlap
distribution of generated samples vs held-out test samples, both measured
against the training set. The KL divergence between these two distributions
(the "nearest-neighbor divergence") quantifies the degree of bias toward the
training data.

Procedure:
1. Load trained model checkpoints across epochs.
2. Sample a fixed starting noise x_T <- N(0, I) once, shared across all epochs.
3. For each epoch:
   a. Generate N_test sequences via deterministic reverse diffusion
      (fix_noise=True) from the shared x_T for cross-epoch comparability.
   b. Compute NN Hamming distance of each generated sequence to training set.
   c. Compute NN Hamming distance of each test sequence to training set
      (baseline / ground-truth reference).
   d. Compare distributions via KL divergence.
   e. Also compute train/test DSM loss for correlation analysis.
4. Plot the NN divergence and DSM test loss vs training epoch.

Interpretation:
- NN divergence minimum before test loss minimum: bias emerges before
  overfitting, identifying the biased generalization phase.
- The histograms show the distribution shift visually at selected epochs.

NOTE: function `compute_nearest_neighbor_overlap` returns Hamming DISTANCE
(number of mismatches), not overlap. Lower values = more similar.
Historically named "overlap" but semantically it's a distance metric.

Usage (run from scripts/ directory)::

    # 15-seed averaged NN divergence (Fig. 4(a), n=12k)
    python nn_divergence.py \\
        --output-root ../plots/ \\
        --paths-gen "<base>/models_restricted_continuous_0/<sub>,...,<base>/models_restricted_continuous_14/<sub>" \\
        --N-test 50000

    where <sub> = test_model_script_30000_12000_cross_entropy_loss/8_512

    Epoch selection: Fig. 4(a) scans all checkpoints. The output identifies
    the NN-divergence minimum and test-loss minimum, which are then used as
    reference epochs for uturn_overlap.py (Fig. 5).
"""
import contextlib
import logging
from argparse import ArgumentParser
from pathlib import Path
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple, Optional
import matplotlib.pyplot as plt

import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.noise_schedules import alpha_bars_schedule, get_alpha_beta_from_alpha_bar
from modules.testing import generate_sequences
from modules.diffusion import forward_process
from modules.losses import compute_cross_entropy_loss
from utils import (
    load_params,
    load_data,
    setup_device,
    create_model,
    load_checkpoint,
    compute_nearest_neighbor_overlap,
    find_available_epochs,
    compute_kl_divergence_np,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compare_test_loss_with_overlap(
    losses: Tuple[np.ndarray, ...],
    overlaps: np.ndarray,
    epochs: np.ndarray,
    save_path: Path,
    start_epoch: Optional[int] = None,
    end_epoch: Optional[int] = None,
    standard_errors_losses: Optional[Tuple[np.ndarray, ...]] = None,
    standard_errors_overlaps: Optional[np.ndarray] = None,
) -> None:
    """Dual-axis plot of NN overlap (left) and test/train loss (right) vs epoch.

    Args:
        losses (tuple[np.ndarray, ...]): one or more loss arrays of length
            ``len(epochs)``.
        overlaps (np.ndarray): per-epoch overlap values.
        epochs (np.ndarray): epoch numbers.
        save_path (Path): output image path.
        start_epoch (int, optional): first epoch to plot.
        end_epoch (int, optional): last epoch to plot.
        standard_errors_losses (tuple, optional): SE bands for losses.
        standard_errors_overlaps (np.ndarray, optional): SE band for overlaps.
    """

    # Matplotlib style (local)
    rc = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "font.size": 11,
    }

    with plt.rc_context(rc):
        fig, ax_score = plt.subplots(figsize=(10, 6), constrained_layout=True)

        # Filter epochs range if specified
        epochs = np.asarray(epochs)
        overlaps = np.asarray(overlaps)

        mask = np.ones_like(epochs, dtype=bool)
        if start_epoch is not None:
            mask &= (epochs >= start_epoch)
        if end_epoch is not None:
            mask &= (epochs <= end_epoch)

        x = epochs[mask]
        score = overlaps[mask]

        # Handle score SE if provided
        score_se = None
        if standard_errors_overlaps is not None:
            se_full = np.asarray(standard_errors_overlaps, dtype=float)
            score_se = se_full[mask]

        # Prepare loss arrays
        n_losses = len(losses)
        if n_losses < 1:
            raise ValueError("`losses` must contain at least one array (test loss).")

        if n_losses == 1:
            train_loss = None
            test_loss = np.asarray(losses[0], dtype=float)[mask]
        else:
            train_loss = np.asarray(losses[0], dtype=float)[mask]
            test_loss = np.asarray(losses[1], dtype=float)[mask]

        # Handle loss SE if provided
        train_se = None
        test_se = None
        if standard_errors_losses is not None:
            se_losses_full = tuple(np.asarray(arr, dtype=float) for arr in standard_errors_losses)
            if n_losses == 1:
                if len(se_losses_full) >= 1:
                    test_se = se_losses_full[0][mask]
            else:
                if len(se_losses_full) >= 1:
                    train_se = se_losses_full[0][mask]
                if len(se_losses_full) >= 2:
                    test_se = se_losses_full[1][mask]

        # Plot score/distance on left axis
        if score_se is not None:
            h_score = ax_score.errorbar(
                x,
                score,
                yerr=score_se,
                fmt="o-",
                markersize=4,
                linewidth=1.8,
                capsize=3,
                capthick=1.2,
                label="Score / distance (lower is better)",
            )
            score_handle = h_score[0]
        else:
            (score_handle,) = ax_score.plot(
                x, score, "o-", markersize=4, linewidth=1.8, label="Score / distance (lower is better)"
            )

        ax_score.set_xlabel("Epoch")
        ax_score.set_ylabel("Score / distance (lower is better)")
        ax_score.grid(True, which="major")

        # Losses on right axis
        ax_loss = ax_score.twinx()
        ax_loss.spines["top"].set_visible(False)

        loss_handles = []
        loss_labels = []

        if train_loss is not None:
            if train_se is not None:
                h_tr = ax_loss.errorbar(
                    x,
                    train_loss,
                    yerr=train_se,
                    fmt="s--",
                    markersize=4,
                    linewidth=1.6,
                    capsize=3,
                    capthick=1.2,
                    label="Train loss",
                )
                loss_handles.append(h_tr[0])
            else:
                (h_tr_line,) = ax_loss.plot(
                    x, train_loss, "s--", markersize=4, linewidth=1.6, label="Train loss"
                )
                loss_handles.append(h_tr_line)
            loss_labels.append("Train loss")

        if test_se is not None:
            h_te = ax_loss.errorbar(
                x,
                test_loss,
                yerr=test_se,
                fmt="x:",
                markersize=5,
                linewidth=1.6,
                capsize=3,
                capthick=1.2,
                label="Test loss",
            )
            loss_handles.append(h_te[0])
        else:
            (h_te_line,) = ax_loss.plot(
                x, test_loss, "x:", markersize=5, linewidth=1.6, label="Test loss"
            )
            loss_handles.append(h_te_line)
        loss_labels.append("Test loss")

        ax_loss.set_ylabel("Loss")

        # Vertical lines: best (minimum) score, best (minimum) test loss
        vlines = []
        vlabels = []

        if score.size > 0 and np.isfinite(score).any():
            idx_best_score = int(np.nanargmin(score))
            best_epoch_score = x[idx_best_score]
            v_score = ax_score.axvline(best_epoch_score, linestyle="-.", linewidth=1.4, label="Best score (min)")
            vlines.append(v_score)
            vlabels.append("Best score (min)")

        if test_loss.size > 0 and np.isfinite(test_loss).any():
            idx_best_test = int(np.nanargmin(test_loss))
            best_epoch_test = x[idx_best_test]
            v_test = ax_score.axvline(best_epoch_test, linestyle="--", linewidth=1.4, label="Best test loss (min)")
            vlines.append(v_test)
            vlabels.append("Best test loss (min)")

        # Title
        fig.suptitle("Training/Test loss vs Score/Distance across epochs")

        # Legend: combine both axes + vlines
        h1, l1 = ax_score.get_legend_handles_labels()
        h2, l2 = ax_loss.get_legend_handles_labels()
        ax_score.legend(h1 + h2 + vlines, l1 + l2 + vlabels, loc="upper right")

        # Optional: log scales
        def _safe_log_scale(arr: np.ndarray) -> bool:
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            return arr.size > 0 and np.all(arr > 0) and (arr.max() / arr.min() >= 100)

        if x.size > 0 and np.all(x > 0) and (x.max() / x.min() >= 100):
            ax_score.set_xscale("log")

        if _safe_log_scale(score):
            ax_score.set_yscale("log")
        if _safe_log_scale(test_loss) or (train_loss is not None and _safe_log_scale(train_loss)):
            ax_loss.set_yscale("log")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)


def plot_nn_overlap_histogram(
    nn_model: np.ndarray,
    nn_bp: np.ndarray,
    seq_len: int,
    save_path: Path,
    n_model: int,
    n_bp: int,
    title: Optional[str] = None,
    n_bins: Optional[int] = None,
) -> None:
    """
    Overlap histogram comparison.
    Uses density=True so the two distributions are comparable even if N differs.
    """
    rc = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "font.size": 11,
    }

    nn_model = np.asarray(nn_model)
    nn_bp = np.asarray(nn_bp)

    if title is None:
        title = "Nearest-neighbor overlap distribution"
    if n_bins is None:
        n_bins = seq_len

    # Bin edges based on data range
    data_min = min(nn_model.min() if nn_model.size else 0, nn_bp.min() if nn_bp.size else 0)
    data_max = max(nn_model.max() if nn_model.size else 1, nn_bp.max() if nn_bp.size else 1)

    # Avoid degenerate binning if all values are identical
    if np.isfinite(data_min) and np.isfinite(data_max) and data_max == data_min:
        data_max = data_min + 1.0

    bins = np.linspace(data_min, data_max, n_bins + 1)

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        ax.hist(
            nn_model,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            linewidth=1.2,
            label=f"Model samples (N={n_model})",
        )
        ax.hist(
            nn_bp,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            linewidth=1.2,
            label=f"Data baseline (N={n_bp})",
        )

        ax.set_title(title)
        ax.set_xlabel("Nearest-neighbor overlap")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)


def main(args):
    """Entry point for NN KL divergence experiment (Exp 2 / Fig 4).

    Loads checkpoints across epochs, computes KL(BP || NN) and
    nearest-neighbour overlap, and plots the joint evolution.

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

    if len(paths) > 1:
        logger.info("Processing multiple model directories: %s", paths)
        # Keep train/test loss and KL divergence containers to keep track of these quantities across models
        test_losses = []
        train_losses = []
        kl_divergences = []
        # Also keep processed epochs per model, to avoid epoch/loss/KL misalignment when some epochs are skipped
        epochs_per_model = []

    # Big loop over model paths
    for i_model, model_dir in enumerate(paths):
        logger.info("Processing %s", model_dir)

        # Load params & data
        params = load_params(model_dir / "full_params.json")
        reduced_length = params["reduced_length"]  # size of the training set
        t_final = params["t_final"]
        pick_i = int(params.get("pick_i_for_training", 0) or 0)  # selection of training set index identifier
        li = pick_i * reduced_length
        ri = li + reduced_length
        vocab_size, sequences, seq_len, _, _, _ = load_data(params["data_path"], device)
        train_seqs = sequences[li:ri]

        # For the test set we take the following N_test (after the rightmost training set index bound)
        test_seqs = sequences[ri : ri + args.N_test]

        # Convert train and test data to one-hot vectors
        train_embeddings = torch.nn.functional.one_hot(train_seqs.long(), num_classes=vocab_size).float().to(device)
        test_embeddings = torch.nn.functional.one_hot(test_seqs.long(), num_classes=vocab_size).float().to(device)

        # Map training sequences to numpy
        train_seqs_np = train_seqs.long().cpu().numpy()

        # Baseline: BP-seq from data
        x0_bp = test_seqs.long().cpu().numpy()

        # For each test sequence, compute NN Hamming distance to training set
        # NOTE: compute_nearest_neighbor_overlap returns Hamming DISTANCE (lower = more similar)
        nn_bp_all = [compute_nearest_neighbor_overlap(x0_bp[i], train_seqs_np)[0] for i in range(len(x0_bp))]

        # Model setup
        model, in_channels, _ = create_model(params, vocab_size, seq_len, device)
        if os.environ.get("NO_COMPILE", "0") != "1":
            model = torch.compile(model)
        else:
            logger.info("Skipping torch.compile (NO_COMPILE=1)")
        logger.info("Model instantiated")

        # Diffusion schedule
        t_test = t_final
        alpha_bars = alpha_bars_schedule(
            t_test, device=device, s=params["s"], schedule="linear"
        )
        alphas, _ = get_alpha_beta_from_alpha_bar(alpha_bars)

        if args.which_t is not None:
            t = int(args.which_t)
            timesteps_tr = torch.full((train_embeddings.shape[0],), t, device=device, dtype=torch.long)
            timesteps = torch.full((test_embeddings.shape[0],), t, device=device, dtype=torch.long)
        else:
            timesteps_tr = torch.randint(
                1, t_final + 1, (train_embeddings.shape[0],), device=device, dtype=torch.long
            )
            timesteps = torch.randint(1, t_final + 1, (test_embeddings.shape[0],), device=device, dtype=torch.long)

        # Forward noising of train and test data
        x_tr = []
        for i, t in tqdm(enumerate(timesteps_tr)):
            x_tr.append(forward_process(train_embeddings[i], t, alpha_bars=alpha_bars))
        x_tr = torch.stack(x_tr)

        x_test = []
        for i, t in tqdm(enumerate(timesteps)):
            x_test.append(forward_process(test_embeddings[i], t, alpha_bars=alpha_bars))
        x_test = torch.stack(x_test)

        # Epochs selection
        epochs = ([args.which_epoch] if args.which_epoch is not None else find_available_epochs(model_dir))
        epochs = sorted(list(set(epochs)))
        if args.max_epoch is not None:
            epochs = [e for e in epochs if e <= args.max_epoch]
        if args.start_epoch is not None:
            epochs = [e for e in epochs if e >= args.start_epoch]
        if not epochs:
            logger.warning("No epochs found in %s", model_dir)
            continue
        logger.info("Found epochs: %s", epochs)

        # Containers for KL divergences and losses for the specific path under consideration
        kls_all = []
        loss_tr = []
        loss_test = []

        # Epoch-aligned containers (prevents plotting errors if some epochs are skipped)
        processed_epochs = []
        loss_tr_aligned = []
        loss_test_aligned = []
        kls_aligned = []

        nbins = args.nbins
        logger.info("Number of bins for histogram: %d", nbins)

        _plots_base = Path(args.output_root) if args.output_root else PROJECT_ROOT / "plots"
        plots_dir = (
            _plots_base
            / "exp2"
            / f"N_train_{params['reduced_length']}"
            / f"N_test_{args.N_test}"
            / f"pick_i_{pick_i}"
            / f"seed_{params['seed']}"
        )
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Starting point is shared across epochs and across models.
        if i_model == 0:
            xt = torch.randn(args.N_test, seq_len, in_channels, device=device)

        # Loop over checkpoints
        for epoch in epochs:
            # define per-epoch loss scalars (filled in cache or recompute branch) for aligned arrays
            loss_tr_epoch = None
            loss_test_epoch = None

            # Check if the file exists in plots_dir
            cache_file = f"nn_overlap_data_epochs_N_test_{args.N_test}_epoch_{epoch}.pkl"
            if cache_file in os.listdir(plots_dir):
                out_nn = plots_dir / cache_file
                with open(out_nn, "rb") as f:
                    data = pickle.load(f)

                nn_bp_all = data["nn_bp_all"]
                nn_model_all = data["nn_model_all"]
                
                # Handle both old format (accumulated lists) and new format (per-epoch scalars)
                # New format uses 'train_loss_epoch' and 'test_loss_epoch' keys
                if "train_loss_epoch" in data:
                    # New format: per-epoch scalar values
                    loss_tr_epoch = float(data["train_loss_epoch"])
                    loss_test_epoch = float(data["test_loss_epoch"])
                else:
                    # Old format: accumulated lists - cannot reliably extract per-epoch value
                    # Skip this epoch and recompute
                    logger.warning(
                        "Cache file %s uses old format (accumulated lists). "
                        "Delete cache to recompute with new format.", cache_file
                    )
                    # Set to None so it falls through to recompute or gets skipped
                    loss_tr_epoch = None
                    loss_test_epoch = None

                logger.info("Loaded nearest-neighbor overlap data from %s", out_nn)

            else:
                # Otherwise, recompute and save to cache for future use
                logger.info("Epoch %d", epoch)
                ckpt = model_dir / f"test_model_script_epoch_{epoch}.pt"
                if not ckpt.exists() or not load_checkpoint(model, str(ckpt), device, training=False):
                    logger.warning("Checkpoint %d failed", epoch)
                    continue
                model.eval()

                logger.info("Computing Losses...")
                
                # Context manager for disabling Flash Attention if requested (workaround for HPC CUDA errors)
                ctx = contextlib.nullcontext()
                if os.environ.get("NO_FLASH", "0") == "1":
                    ctx = torch.nn.attention.sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
                
                with torch.no_grad(), torch.amp.autocast('cuda'), ctx:
                    x0_hat_test = model(x_test, timesteps)
                    x0_hat_tr = model(x_tr, timesteps_tr)
                x0_hat_test = x0_hat_test.float()
                x0_hat_tr = x0_hat_tr.float()

                loss_test.append(
                    compute_cross_entropy_loss(test_embeddings.argmax(-1), x0_hat_test).mean().cpu().item()
                )
                loss_tr.append(
                    compute_cross_entropy_loss(train_embeddings.argmax(-1), x0_hat_tr).mean().cpu().item()
                )
                logger.info("Losses computed successfully")

                # Record per-epoch loss scalars for aligned arrays
                loss_tr_epoch = float(loss_tr[-1])
                loss_test_epoch = float(loss_test[-1])

                # Sample sequences
                logger.info("Generating Sequences...")
                x0_hat = generate_sequences(
                    args.N_test,
                    model,
                    t_test,
                    alpha_bars,
                    alphas,
                    seq_len,
                    in_channels,
                    device,
                    clamp=False,
                    temperature=None,
                    batch_size=args.batch_size if args.batch_size else args.N_test,
                    bp_params=None,
                    start_seqs=xt,  # NOTE: we are fixing the starting point for better comparability
                    fix_noise=True,  # deterministic x_T -> x_0 mapping for cross-epoch comparability
                ).cpu()
                logger.info("Sequences generated successfully")

                logger.info("Computing nearest-neighbor overlap for all samples")
                x0_hat = x0_hat.numpy()
                # Compute NN Hamming distance of each generated sample to training set
                # Lower distance = generated sample is more similar to training data (bias signal)
                nn_model_all = [compute_nearest_neighbor_overlap(x0_hat[i], train_seqs_np)[0] for i in range(len(x0_hat))]
                nn_bp_all = np.array(nn_bp_all)
                nn_model_all = np.array(nn_model_all)

                out_all = plots_dir / f"nn_overlap_all_epoch_{epoch}.png"
                plot_nn_overlap_histogram(
                    nn_model_all,
                    nn_bp_all,
                    seq_len,
                    out_all,
                    n_model=len(nn_model_all),
                    n_bp=len(nn_bp_all),
                    title="Nearest‐Neighbor Overlap (All Samples)",
                    n_bins=nbins,
                )

                # Save all NN overlap data with per-epoch scalar losses (new format)
                out_nn = plots_dir / f"nn_overlap_data_epochs_N_test_{args.N_test}_epoch_{epoch}.pkl"
                with open(out_nn, "wb") as f:
                    pickle.dump(
                        {
                            "nn_bp_all": nn_bp_all,
                            "nn_model_all": nn_model_all,
                            "train_loss_epoch": loss_tr_epoch,  # per-epoch scalar
                            "test_loss_epoch": loss_test_epoch,  # per-epoch scalar
                        },
                        f,
                    )
                logger.info("Nearest-neighbor overlap data saved to %s", out_nn)

            if loss_tr_epoch is None or loss_test_epoch is None:
                logger.warning("Missing per-epoch losses for epoch %d; skipping epoch for aligned plots.", epoch)
                continue

            # Bin the counts
            data_min = 0
            data_max = seq_len
            # With this logic, the support of the two distribution is consistent and always the same
            bins = np.linspace(data_min, data_max, nbins + 1)
            counts_gen, _ = np.histogram(nn_model_all, bins=bins)
            counts_bp, _ = np.histogram(nn_bp_all, bins=bins)

            probs_gen = counts_gen / counts_gen.sum() if counts_gen.sum() > 0 else np.zeros_like(counts_gen)
            probs_bp = counts_bp / counts_bp.sum() if counts_bp.sum() > 0 else np.zeros_like(counts_bp)

            if args.distance_metric == "kl":
                kl_gen = compute_kl_divergence_np(probs_gen, probs_bp)
                logger.info("KL divergence Gen: %.4f at epoch %d", kl_gen, epoch)
            elif args.distance_metric == "euclidean":
                kl_gen = np.linalg.norm(probs_gen - probs_bp)
                logger.info("Euclidean distance Gen: %.4f at epoch %d", kl_gen, epoch)
            else:
                raise ValueError(f"Unknown distance metric: {args.distance_metric}")

            kls_all.append(kl_gen)
            processed_epochs.append(epoch)
            kls_aligned.append(float(kl_gen))
            loss_tr_aligned.append(float(loss_tr_epoch))
            loss_test_aligned.append(float(loss_test_epoch))

        out_compare_kl = plots_dir / f"kl_divergence_overlap_score_epochs_N_test_{args.N_test}.png"
        compare_test_loss_with_overlap(
            losses=(np.array(loss_tr_aligned), np.array(loss_test_aligned)),
            overlaps=np.array(kls_aligned),
            epochs=np.array(processed_epochs),
            save_path=out_compare_kl,
        )

        # Save per-model results as .npz for offline plotting
        out_npz = plots_dir / f"exp2_results_N_test_{args.N_test}.npz"
        np.savez(
            out_npz,
            epochs=np.array(processed_epochs),
            kl_divergences=np.array(kls_aligned),
            train_losses=np.array(loss_tr_aligned),
            test_losses=np.array(loss_test_aligned),
        )
        logger.info("Saved per-model results to %s", out_npz)

        if len(paths) > 1:
            epochs_per_model.append(list(processed_epochs))
            kl_divergences.append(np.array(kls_aligned))
            test_losses.append(np.array(loss_test_aligned))
            train_losses.append(np.array(loss_tr_aligned))

    if len(paths) > 1:
        common_epochs = set(epochs_per_model[0]) if epochs_per_model else set()
        for ep_list in epochs_per_model[1:]:
            common_epochs &= set(ep_list)
        common_epochs = sorted(common_epochs)

        if not common_epochs:
            logger.warning("No common processed epochs across models; skipping averaged plot.")
            return

        # Build aligned matrices [n_models, n_common_epochs]
        kl_mat = []
        tr_mat = []
        te_mat = []

        for ep_list, kl_arr, tr_arr, te_arr in zip(epochs_per_model, kl_divergences, train_losses, test_losses):
            idx = {e: i for i, e in enumerate(ep_list)}
            sel = [idx[e] for e in common_epochs]
            kl_mat.append(kl_arr[sel])
            tr_mat.append(tr_arr[sel])
            te_mat.append(te_arr[sel])

        kl_mat = np.stack(kl_mat, axis=0)
        tr_mat = np.stack(tr_mat, axis=0)
        te_mat = np.stack(te_mat, axis=0)

        kl_mean = np.mean(kl_mat, axis=0)
        kl_se = np.std(kl_mat, axis=0) / np.sqrt(kl_mat.shape[0])

        tr_mean = np.mean(tr_mat, axis=0)
        tr_se = np.std(tr_mat, axis=0) / np.sqrt(tr_mat.shape[0])

        te_mean = np.mean(te_mat, axis=0)
        te_se = np.std(te_mat, axis=0) / np.sqrt(te_mat.shape[0])

        _plots_base2 = Path(args.output_root) if args.output_root else PROJECT_ROOT / "plots"
        plots_dir_general = (
            _plots_base2
            / "exp2_averaged"
            / f"N_train_{params['reduced_length']}"
            / f"N_test_{args.N_test}"
            / f"pick_i_{pick_i}"
            / f"seed_{params['seed']}"
        )
        plots_dir_general.mkdir(parents=True, exist_ok=True)  

        out_compare_kl = plots_dir_general / f"AVERAGED_kl_divergence_overlap_score_epochs_N_test_{args.N_test}.png"
        compare_test_loss_with_overlap(
            (tr_mean, te_mean),
            np.array(kl_mean),
            np.array(common_epochs),
            out_compare_kl,
            None,
            None,
            (tr_se, te_se),
            kl_se,
        )

        # Save multi-model aggregated results as .npz for offline plotting
        out_npz_agg = plots_dir_general / f"exp2_aggregated_N_test_{args.N_test}.npz"
        np.savez(
            out_npz_agg,
            epochs=np.array(common_epochs),
            kl_mean=kl_mean,
            kl_se=kl_se,
            train_loss_mean=tr_mean,
            train_loss_se=tr_se,
            test_loss_mean=te_mean,
            test_loss_se=te_se,
        )
        logger.info("Saved aggregated results to %s", out_npz_agg)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--paths-gen",
        type=str,
        default="../results_transformer_pick_0_for_training/models_restricted_continuous_0/test_model_script_20000_5000_cross_entropy_loss/8_512",
        help="Comma-separated list of model generation directories",
    )
    p.add_argument("--N-test", type=int, default=50000)
    p.add_argument("--which-epoch", type=int, default=None,
                   help="Evaluate a single epoch. Omit to scan all available "
                        "checkpoints (needed to locate the NN-divergence "
                        "minimum and the test-loss minimum).")
    p.add_argument("--start-epoch", type=int, default=None,
                   help="Discard checkpoints before this epoch.")
    p.add_argument("--max-epoch", type=int, default=None,
                   help="Discard checkpoints after this epoch.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--which_t", type=int, default=None)
    p.add_argument("--nbins", type=int, default=20, help="Number of bins for histogram")
    p.add_argument(
        "--distance-metric",
        type=str,
        default="kl",
        choices=["euclidean", "kl"],
        help="Distance metric to use for nearest neighbor overlap",
    )
    p.add_argument("--output-root", type=str, default=None,
                   help="Root dir for output plots (default: PROJECT_ROOT/plots)")
    args = p.parse_args()
    main(args)
