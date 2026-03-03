"""
Utility functions for training and testing diffusion models.
Includes plotting, testing, checkpointing, and model/data loading functions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, List    
import logging
import json
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt
import seaborn as sns

from typing import Optional, Any

from modules.transformer_models import TransformerForDiffusion

###############################################################################
##### TEST METRICS ############################################################
###############################################################################


def compute_kl_divergence(P, Q, eps=1e-8):
    """Compute KL(P || Q) element-wise with clamping for numerical safety.

    Args:
        P (torch.Tensor): reference distribution.
        Q (torch.Tensor): approximate distribution (same shape as *P*).
        eps (float): clamping floor.

    Returns:
        torch.Tensor: scalar KL divergence.
    """
    P_safe = torch.clamp(P, min=eps)  # Avoid log(0)
    Q_safe = torch.clamp(Q, min=eps)  # Avoid division by zero
    return torch.sum(P_safe * torch.log(P_safe / Q_safe))

def compute_kl_divergence_np(P, Q, eps=1e-8):
    """Compute KL(P || Q) with NumPy, averaging over the batch dim if 2-D.

    Args:
        P (np.ndarray): reference distribution (1-D or 2-D).
        Q (np.ndarray): approximate distribution (same shape).
        eps (float): clamping floor.

    Returns:
        float: KL divergence (summed over last axis, averaged over batch).
    """
    P_safe = np.clip(P, a_min=eps, a_max = 1)  # Avoid log(0)
    Q_safe = np.clip(Q, a_min=eps, a_max = 1)  # Avoid division by zero
    temp = P_safe * np.log(P_safe / Q_safe)
    if len(temp.shape) == 1:
        return np.sum(temp)
    elif len(temp.shape) == 2:
       return np.mean(np.sum(temp, axis = -1))
    else:
        raise Exception

def compute_dot_overlap(P, Q):
    """Compute normalised dot product between P and Q.

    Args:
        P (np.ndarray): first array (1-D or 2-D).
        Q (np.ndarray): second array (same shape).

    Returns:
        float: dot product (normalised by batch size if 2-D).
    """
    if len(P.shape) == 1:
        return np.dot(P.flatten(),Q.flatten())
    elif len(P.shape) == 2:
        return np.dot(P.flatten(),Q.flatten())/P.shape[0]
    else:
        raise ValueError(f"compute_dot_overlap: expected 1-D or 2-D arrays, got {P.ndim}-D")


def compute_nearest_neighbor_overlap(seq, train_seqs):
    """
    Compute Hamming distance to nearest neighbor in training set.
    
    NOTE: Despite the name, this returns DISTANCE (# of mismatches), not overlap.
    Lower values = more similar to training set.
    
    Args:
        seq: Single sequence of shape (L,)
        train_seqs: Training sequences of shape (N, L)
    
    Returns:
        (min_distance, argmin_idx): Hamming distance to nearest neighbor and its index
    """
    diffs = np.count_nonzero(train_seqs != seq, axis=1)
    idx = np.argmin(diffs)
    return diffs[idx], idx


def kl_divergence_batch(
    P: np.ndarray, Q: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Vectorised per-sample KL divergence.

    Args:
        P (np.ndarray): reference distributions, shape ``(N, L, Q)``.
        Q (np.ndarray): approximate distributions, same shape.
        eps (float): clamping floor.

    Returns:
        np.ndarray: per-sample KL, shape ``(N,)``.
    """
    P_safe = np.clip(P, eps, None)
    Q_safe = np.clip(Q, eps, None)
    kl_per_token = np.sum(P_safe * np.log(P_safe / Q_safe), axis=-1)
    return np.sum(kl_per_token, axis=-1)


def dot_overlap_batch(
    P: np.ndarray, Q: np.ndarray
) -> np.ndarray:
    """Vectorised per-sample dot overlap.

    Args:
        P (np.ndarray): first array, shape ``(N, L, Q)``.
        Q (np.ndarray): second array, same shape.

    Returns:
        np.ndarray: per-sample dot product, shape ``(N,)``.
    """
    N = P.shape[0]
    flat_P = P.reshape(N, -1)
    flat_Q = Q.reshape(N, -1)
    return np.einsum('ij,ij->i', flat_P, flat_Q)


def nearest_neighbor_overlap_batch(
    seqs: np.ndarray, train_seqs: np.ndarray
) -> np.ndarray:
    """Per-sample Hamming distance to nearest training sequence.

    Args:
        seqs (np.ndarray): test sequences, shape ``(N_test, L)``.
        train_seqs (np.ndarray): training sequences, shape ``(N_train, L)``.

    Returns:
        np.ndarray: minimum Hamming distances, shape ``(N_test,)``.
    """
    # distances: [N_train, N_test]
    dists = np.sum(train_seqs[:, None, :] != seqs[None, :, :], axis=2)
    return np.min(dists, axis=0)


###############################################################################
##### TRAINING FUNCTIONS ######################################################
###############################################################################

def get_checkpoint_files(checkpoint_dir):
    """
    Return a sorted list of checkpoint files in the given directory.
    """
    files = [f for f in os.listdir(checkpoint_dir) if 'test_model_script' in f]
    return sorted(files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))


def create_save_path(data_path, n_iter, reduced_length, loss_name, num_layers, hidden_size, discrete_diffusion, rescaled, seed=None, model_type="transformer", add_flag=None, results_dir=None):
    """
    Create and return the directory to save models based on data path and parameters.
    """
    # determine dataset tag from the data path
    if 'restricted' in data_path:
        dataset_tag = 'restricted'
    elif 'uniform' in data_path:
        dataset_tag = 'uniform'
    else:
        dataset_tag = 'standard'

    diff_tag = 'continuous' if not discrete_diffusion else 'discrete'
    rescaled_tag = '_rescaled' if rescaled else ''
    dir_name = f'models_{dataset_tag}_{diff_tag}{rescaled_tag}_{seed}'

    if results_dir is not None:
        base_dir = os.path.join(results_dir, f"results_{model_type}", dir_name)
    else:
        base_dir = f'../results_{model_type}/{dir_name}'

    save_path = os.path.join(
        base_dir,
        f"test_model_script_{n_iter}_{reduced_length}_{loss_name}",
        f"{num_layers}_{hidden_size}"
    )
    if add_flag is not None:
         save_path = save_path + "_" + add_flag
         
    os.makedirs(save_path, exist_ok=True)
    return save_path



###############################################################################
##### GENERAL FUNCTIONS #######################################################
###############################################################################

def find_available_epochs(model_dir: Path) -> list[int]:
    """Discover saved checkpoint epochs in *model_dir*.

    Args:
        model_dir (Path): directory containing ``test_model_script_epoch_*.pt`` files.

    Returns:
        list[int]: sorted epoch numbers found.
    """
    epochs = []
    for ckpt in model_dir.glob("test_model_script_epoch_*.pt"):
        try:
            epochs.append(int(ckpt.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return sorted(epochs)


def load_checkpoint(model: nn.Module, 
                    checkpoint_path: str = None, 
                    checkpoint_dir: str = None, 
                    optimizer: Optional[torch.optim.Optimizer] = None, 
                    scheduler: Optional[Any] = None, 
                    device: torch.device = None,
                    training: bool = False):
    """Load a checkpoint and restore model (and optionally optimiser/scheduler) state.

    Handles ``torch.compile`` prefix mismatches and legacy ``nn.Sequential``
    projection layers automatically.

    Args:
        model (nn.Module): model whose weights to restore.
        checkpoint_path (str, optional): explicit path to a ``.pt`` file.
        checkpoint_dir (str, optional): directory to auto-select the latest
            checkpoint from (used when *checkpoint_path* is None).
        optimizer (Optimizer, optional): optimiser to restore.
        scheduler (optional): LR scheduler to restore.
        device (torch.device, optional): map location for loading.
        training (bool): if True, return ``(epoch, counter, path)``;
            otherwise return a bool success flag.

    Returns:
        If *training* is True: ``tuple[int, int, str]``.
        Otherwise: ``bool`` indicating load success.
    """
    try:
        # Determine checkpoint path.
        if checkpoint_path is None:
            if checkpoint_dir is None:
                raise ValueError("Either checkpoint_path or checkpoint_dir must be provided.")
            checkpoint_files = get_checkpoint_files(checkpoint_dir)
            if not checkpoint_files:
                raise ValueError("No checkpoint files found in the specified directory.")
            latest_ckpt = checkpoint_files[-1]
            ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
        else:
            ckpt_path = checkpoint_path
            latest_ckpt = os.path.basename(ckpt_path)
        
        print("Loading model from checkpoint:", ckpt_path)
        # Load checkpoint with proper map_location.
        if device is not None and device.type == 'cpu':
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(ckpt_path)
        
        # Handle torch.compile() prefix mismatch between model and checkpoint
        state_dict = checkpoint['model_state_dict']
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # Check if checkpoint has _orig_mod. prefix but model doesn't
        ckpt_has_prefix = any(k.startswith('_orig_mod.') for k in ckpt_keys)
        model_has_prefix = any(k.startswith('_orig_mod.') for k in model_keys)
        
        if ckpt_has_prefix and not model_has_prefix:
            # Strip prefix from checkpoint keys
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        elif not ckpt_has_prefix and model_has_prefix:
            # Add prefix to checkpoint keys (model is compiled, checkpoint is not)
            state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
            
        # Handle backward compatibility: remove '.0.' from projection layers that were changed from nn.Sequential(nn.Linear) to nn.Linear
        patched_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for proj in ['input_up_projection', 'time_embedding', 'output_down_projection']:
                if f'{proj}.0.' in new_k:
                    new_k = new_k.replace(f'{proj}.0.', f'{proj}.')
            patched_state_dict[new_k] = v
        state_dict = patched_state_dict
        
        # Load model state.
        model.load_state_dict(state_dict)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if training:
            try:
                epoch_start = int(latest_ckpt.split('_')[-1][:-3])
            except Exception:
                epoch_start = None
            counter_start = checkpoint.get('count_fun', 0)
            return epoch_start, counter_start, ckpt_path
        else:
            return True
    except FileNotFoundError:
        print(f"Checkpoint file not found at {checkpoint_path if checkpoint_path else ckpt_path}.")
        return False if not training else (None, None, None)

def load_params(params_path: str) -> dict:
    """
    Load model parameters from a JSON file.
    """
    with open(params_path, 'r') as f:
        return json.load(f)

def setup_device():
    """
    Setup and return the appropriate torch device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(data_path, device):
    """Load controlled setting data from ``.npy``, auto-generating if the file is missing.

    Args:
        data_path (str): path to the ``.npy`` data file.
        device (torch.device): target device for loaded tensors.

    Returns:
        tuple: ``(vocab_size, sequences, seq_len, k, rho, data_path)``.
    """
    if not os.path.exists(data_path):
        import re
        filename = os.path.basename(data_path)
        # Parse format: prefix_Q_L_SIGMA_QEFF_SEED.npy
        match = re.search(r'_(\d+)_(\d+)_([\d\.]+)_(\d+)_(\d+)\.npy$', filename)
        if match:
            q = int(match.group(1))
            l = int(match.group(2))
            sigma = float(match.group(3))
            q_eff = int(match.group(4))
            seed = int(match.group(5))
            print(f"Data file not found. Auto-generating data with q={q}, l={l}, sigma={sigma}, q_eff={q_eff}, seed={seed}...")
            
            # Import generator
            import sys
            from pathlib import Path
            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
                
            from modules.gen_filtered_hierarchical_data_wforbidden import generate_dataset
            data = generate_dataset(q, l, sigma, q_eff, seed)
            
            os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
            np.save(data_path, data)
            print(f"Successfully generated and saved data to {data_path}")
        else:
            raise FileNotFoundError(f"{data_path!r} does not exist and its name does not match the auto-generation pattern.")

    q, k, _, _, leaves, rho, _ = np.load(data_path, allow_pickle=True)
    vocab_size = q
    sequences = torch.from_numpy(leaves.copy().T).to(torch.int).to(device)
    seq_len = sequences.shape[-1]
    return vocab_size, sequences, seq_len, k, rho, data_path



def initialize_model(vocab_size, seq_len, hidden_size_encoder, model_channels,
                     encoder_model, device):
    """
    Initialize and return the diffusion transformer model.
    """
    in_channels = vocab_size
    out_channels = vocab_size
    model = TransformerForDiffusion(
        seq_len=seq_len,
        in_channels=in_channels,
        hidden_size=hidden_size_encoder,
        vocab_size=vocab_size,
        model_channels=model_channels,
        out_channels=out_channels,
        encoder=encoder_model,
        device=device
    ).to(device)
    return model


def create_model(params: dict, vocab_size: int, seq_len: int, device: torch.device):
    """Instantiate a ``TransformerForDiffusion`` from a parameter dict.

    Args:
        params (dict): model hyper-parameters (as saved in ``full_params.json``).
        vocab_size (int): alphabet size.
        seq_len (int): sequence length.
        device (torch.device): target device.

    Returns:
        tuple: ``(model, in_channels, hidden_size_encoder)``.
    """
    hidden_size_encoder = params['hidden_size_encoder']
    nhead = params['nhead']
    num_layers = params['num_layers']
    model_channels = params['model_channels']

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_size_encoder,
        nhead=nhead,
        dim_feedforward=2 * hidden_size_encoder,
        batch_first=True
    )
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    # Update hidden size if necessary.
    hidden_size_encoder_ = transformer_encoder.layers[0].self_attn.embed_dim
    assert hidden_size_encoder == hidden_size_encoder_, \
        f"Hidden size mismatch: {hidden_size_encoder} != {hidden_size_encoder_}"

    model = initialize_model(vocab_size, seq_len, hidden_size_encoder, model_channels,
                             transformer_encoder, device)
    
    in_channels = vocab_size

    return model, in_channels, hidden_size_encoder
