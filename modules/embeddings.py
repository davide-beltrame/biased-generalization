import math
import torch


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal positional embedding for diffusion timesteps.

    Produces the same encoding used in "Attention Is All You Need" (Vaswani
    et al.) and adapted for diffusion models as per "Denoising Diffusion Probabilistic Models" (Ho et al.).

    Args:
        timesteps (torch.Tensor): 1-D tensor of timestep indices.
        dim (int): embedding dimension.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        torch.Tensor: embeddings of shape ``(len(timesteps), dim)``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args_time = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args_time), torch.sin(args_time)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
