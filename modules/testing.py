import torch
from tqdm import tqdm

from .diffusion import (
    backward_process,
    backward_process_gt,
    backward_process_gt_torch,
    backward_process_empirical,
    backward_process_prior
)
from .belief_propagation import run_BP


@torch.inference_mode()
def generate_sequences(
    N_test,
    transformer_for_diffusion,
    t_final,
    alpha_bars,
    alphas,
    seq_len,
    in_channels,
    device,
    clamp=False,
    temperature=1.0,
    batch_size=10000,
    bp_params=None,
    start_seqs=None,
    prior_score=False,
    empirical_score=False,
    x0s=None,
    factorized_layers=0,
    return_trajectories_and_scores=False,
    fix_noise=False,
    bp_backend='torch',
):
    """Generate sequences via the backward diffusion process.

    Optionally returns intermediate trajectories and score estimates.

    Args:
        N_test (int): number of test samples to generate.
        transformer_for_diffusion (nn.Module): denoising transformer.
        t_final (int): total number of diffusion steps.
        alpha_bars (torch.Tensor): precomputed alpha-bar schedule.
        alphas (torch.Tensor): precomputed alpha schedule.
        seq_len (int): sequence length.
        in_channels (int): embedding dimension (vocabulary size for one-hot).
        device (torch.device): computation device.
        clamp (bool): if True, clamp the score to one-hot at each step.
        temperature (float): softmax temperature for the score.
        batch_size (int): samples processed per forward batch.
        bp_params (tuple, optional): (M, l, q) for belief-propagation score.
            If None, the neural-network score is used.
        start_seqs (torch.Tensor, optional): initial x_T; random if None.
        prior_score (bool): use the prior (field-only) score.
        empirical_score (bool): use the empirical Bayes score.
        x0s (torch.Tensor, optional): training set, required when
            *empirical_score* is True.
        factorized_layers (int): number of factorized tree levels for BP_k.
        return_trajectories_and_scores (bool): if True, also return
            full trajectories and per-step score estimates.
        fix_noise (bool): use deterministic cross-device noise.
        bp_backend (str): ``'torch'`` (batched GPU) or ``'numpy'``
            (serial CPU).

    Returns:
        torch.Tensor: generated sequences of shape ``(N_test, seq_len)``.
        If *return_trajectories_and_scores* is True, returns a tuple
        ``(x0_pred, scores, trajectories)``.
    """
    T = t_final

    # Initialize x_T
    if start_seqs is None:
        xT = torch.randn(N_test, seq_len, in_channels, device=device)
    else:
        xT = start_seqs.to(device)
        assert xT.shape == (N_test, seq_len, in_channels), \
            f"start_seqs must be (N_test, seq_len, in_channels), got {xT.shape}"

    batched = list(torch.split(xT, batch_size))
    assert sum(b.size(0) for b in batched) == N_test, \
        "Batch splitting error: total samples mismatch N_test."

    x0_preds = []
    if return_trajectories_and_scores:
        scores = torch.zeros(T, N_test, seq_len, in_channels, device=device)
        # Extra slot at index T stores the initial x_T
        trajectories = torch.zeros(T+1, N_test, seq_len, in_channels, device=device)
        trajectories[T, :] = xT.clone()

    offset = 0
    # Process each batch
    for xT_batch in tqdm(batched, desc="Processing batches"):
        bsz = xT_batch.size(0)
        start = offset
        end = start + bsz
        x_curr = xT_batch.clone()

        # Backward diffusion steps
        for t in tqdm(range(T, 0, -1), desc="Processing steps", leave=False):
            if not bp_params:
                if prior_score:
                    x_curr, x0_hat = backward_process_prior(
                        x_curr, t, device,
                        alpha_bars=alpha_bars, alphas=alphas,
                        fix_noise=fix_noise
                    )
                elif empirical_score:
                    x_curr, x0_hat = backward_process_empirical(
                        x_curr, x0s, t, device,
                        alpha_bars=alpha_bars, alphas=alphas,
                        fix_noise=fix_noise
                    )
                else:
                    x_curr, x0_hat = backward_process(
                        x_curr, t, t_final, transformer_for_diffusion,
                        device, alpha_bars=alpha_bars, alphas=alphas,
                        clamp=clamp, temperature=temperature,
                        fix_noise=fix_noise
                    )
            else:
                if bp_backend == 'torch':
                    x_curr, x0_hat = backward_process_gt_torch(
                        x_curr, t, bp_params, device,
                        alpha_bars=alpha_bars, alphas=alphas,
                        factorized_layers=factorized_layers,
                        fix_noise=fix_noise
                    )
                else:
                    x_curr = x_curr.cpu().numpy()
                    x_curr, x0_hat = backward_process_gt(
                        x_curr, t, bp_params, device,
                        alpha_bars=alpha_bars, alphas=alphas,
                        factorized_layers=factorized_layers,
                        fix_noise=fix_noise
                    )

            if return_trajectories_and_scores:
                scores[t-1, start:end] = x0_hat
                trajectories[t-1, start:end] = x_curr

        x_final = x_curr.clone()
        x0_preds.append(torch.argmax(x_final, dim=-1))

        if return_trajectories_and_scores:
            trajectories[0, start:end] = x_final

        offset = end

    assert offset == N_test, \
        f"Processed {offset} samples but expected {N_test}."
    x0_pred = torch.cat(x0_preds, dim=0)
    assert x0_pred.shape[0] == N_test, \
        f"x0_pred has {x0_pred.shape[0]} samples but expected {N_test}."

    if return_trajectories_and_scores:
        return x0_pred, scores, trajectories
    return x0_pred


def compute_bp_free_energies(sequences, M, l, q, backend='torch', batch_size=256):
    """Compute free energies for a batch of sequences using belief propagation.

    Args:
        sequences (array-like): discrete sequences of shape ``(n, seq_len)``.
            Values in ``[0, q-1]`` or ``q+1`` for masked positions.
        M (np.ndarray): transition tensor of shape ``(q, q, q)``.
        l (int): depth of the binary tree (sequence length = ``2^l``).
        q (int): alphabet size.
        backend (str): ``'torch'`` (batched GPU, faster) or ``'numpy'``
            (serial CPU).
        batch_size (int): batch size for the torch backend.

    Returns:
        list[float]: per-sequence free energies. Lower values indicate
        sequences more consistent with the generative model.
    """
    if backend == 'torch':
        from modules.bp_torch import compute_bp_free_energies_torch
        return compute_bp_free_energies_torch(sequences, M, l, q, batch_size=batch_size)
    elif backend == 'numpy':
        free_energies = [
            run_BP(M=M, l=l, q=q, xis=sequence, factorized_layers=0)[1]
            for sequence in tqdm(sequences, desc="Computing BP free energies")
        ]
        return free_energies
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'numpy'.")