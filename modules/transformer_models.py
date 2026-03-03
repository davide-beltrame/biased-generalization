import torch
import torch.nn as nn
from typing import Optional
from .embeddings import timestep_embedding


class TransformerForDiffusion(nn.Module):
    """Denoising transformer for the backward diffusion process.

    Combines input projection, sinusoidal timestep embedding, learned positional
    embedding, a standard encoder-only transformer, and an output projection
    to predict x_0 from x_t.
    """
    def __init__(
        self,
        seq_len : int,
        in_channels : int,
        hidden_size : int,
        vocab_size : int,
        model_channels : int,
        out_channels : int,
        encoder : torch.nn.Module,
        device : torch.device,
        layer_norm_eps: Optional[float]=None,
        dropout_prob: Optional[float]=None
    ):
        """
        Args:
            seq_len (int): fixed sequence length.
            in_channels (int): input embedding dimension.
            hidden_size (int): hidden representation size per token.
            vocab_size (int): vocabulary size (number of symbols).
            model_channels (int): intermediate dimension for timestep embedding.
            out_channels (int): output dimensionality (typically ``vocab_size``).
            encoder (torch.nn.Module): encoder-only transformer block.
            device (torch.device): target device.
            layer_norm_eps (float, optional): epsilon for layer-norm
                (recorded for logging, not used in forward).
            dropout_prob (float, optional): dropout probability
                (recorded for logging, not used in forward).
        """
        super().__init__()

        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.device = device
        self.layer_norm_eps = layer_norm_eps
        self.dropout_prob = dropout_prob

        # Project the input embeddings to the hidden size
        self.input_up_projection = nn.Linear(self.in_channels, self.hidden_size)
        
        # Standard transformer positional embedding 
        self.positional_embedding = nn.Embedding(self.seq_len, self.hidden_size)
        
        # Project the time embeddings to the hidden size
        self.time_embedding = nn.Linear(self.model_channels, self.hidden_size)
        
        if layer_norm_eps is not None:
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = None
        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None
        
        # Encoder model
        self.transformer_encoder = encoder

        # Project the output to the vocab size
        self.output_down_projection = nn.Linear(self.hidden_size, self.out_channels)
        self.to(device)

    def get_params_dict(self):
        """Return a dictionary of model hyperparameters (excluding encoder weights).

        Returns:
            dict: keys include ``seq_len``, ``in_channels``, ``hidden_size``,
            ``vocab_size``, ``model_channels``, ``out_channels``, ``device``,
            ``layer_norm_eps``, ``dropout_prob``, and ``encoder_params``.
        """
        return {
            'seq_len': int(self.seq_len),
            'in_channels': int(self.in_channels),
            'hidden_size': int(self.hidden_size),
            'vocab_size': int(self.vocab_size),
            'model_channels': int(self.model_channels),
            'out_channels': int(self.out_channels),
            'device': str(self.device),
            'layer_norm_eps': self.layer_norm_eps,
            'dropout_prob': self.dropout_prob,
            'encoder_params': {
                'num_layers': self.transformer_encoder.num_layers,
                'nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
                'd_model': self.transformer_encoder.layers[0].self_attn.embed_dim
            }
        }

    def forward(self, x : torch.Tensor, timesteps : torch.Tensor) -> torch.Tensor:   
        """Forward pass: predict x_0 logits from noisy input.

        Args:
            x (torch.Tensor): token embeddings, shape
                ``(batch_size, seq_len, in_channels)``.
            timesteps (torch.Tensor): per-sample timesteps, shape
                ``(batch_size,)``.

        Returns:
            torch.Tensor: output logits, shape
            ``(batch_size, seq_len, out_channels)``.
        """
        batch_size = x.shape[0]
            
        # Project the embeddings to the latent space
        x_projected = self.input_up_projection(x)

        # Compute and project positional embeddings
        positions = torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        positional_embeddings = self.positional_embedding(positions)

        # Compute time embeddings 
        timestep_embeddings = self.time_embedding(timestep_embedding(timesteps, dim=self.model_channels))

        # Combine the embeddings via element-wise addition
        embedded_inputs = (
            x_projected +
            positional_embeddings +
            timestep_embeddings.unsqueeze(1).expand(-1, self.seq_len, -1)
        )

        # Apply layer normalization if specified
        if self.layer_norm is not None:
            embedded_inputs = self.layer_norm(embedded_inputs)
            
        # Apply dropout if specified
        if self.dropout is not None:
            embedded_inputs = self.dropout(embedded_inputs)

        # Pass through the transformer encoder. Assumes batch_first=True
        latent_reps = self.transformer_encoder(embedded_inputs)

        # Project down to the output space
        output = self.output_down_projection(latent_reps)

        return output
