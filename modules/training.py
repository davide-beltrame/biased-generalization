import os
import json
from tqdm import trange
import torch
import wandb
from .losses import compute_mse_loss, compute_cross_entropy_loss, get_loss_coefficients_continuous_new
from .diffusion import generate_noisy_sequences_single


class DiffusionDataset(torch.utils.data.Dataset):
    """
    Dataset for generating noisy sequences for training a diffusion model.
    Args:
        x0s (torch.Tensor): Original sequences (one-hot encoded).
        t_final (int): Final timestep for the diffusion process.
        alpha_bars (torch.Tensor): Schedule of alpha_bar values.
        reweight (bool): Whether to use reweighting in data generation.
        device (str): Device to use for tensor operations.
    """
    def __init__(self, x0s, t_final, alpha_bars,
                 reweight=False, device="cpu"):
        """Initialise the dataset.

        Args:
            x0s (torch.Tensor): clean sequences (one-hot), shape ``(N, L, Q)``.
            t_final (int): number of diffusion steps.
            alpha_bars (torch.Tensor): alpha-bar schedule, length ``t_final``.
            reweight (bool): if True, sample timesteps with Gaussian weight
                centred at ``t/T = 0.15``; otherwise uniform.
            device (str): device for tensor storage.
        """
        self.x0s = x0s.to(device)
        self.t_final = t_final
        self.alpha_bars = alpha_bars.to(device)
        self.reweight = reweight

    def __len__(self):
        """Return the number of training sequences."""
        return self.x0s.shape[0]

    def __getitem__(self, idx):
        """Return ``(x_t, t, x_0)`` for a single sample via online noising."""
        x0 = self.x0s[idx]
        xt = generate_noisy_sequences_single(
            x0=x0,
            t_final=self.t_final,
            alpha_bars=self.alpha_bars,
            reweight=self.reweight,
            device=self.x0s.device,
        )
        return xt


def training_step(
    model,
    xts_batch,
    x0s_batch,
    timesteps_batch,
    optimizer,
    use_compute_mse_loss,
    use_cross_entropy_loss,
    rescale_loss=False,
    alpha_bars=None,
    alphas=None,
    scaler=None,
):
    """
    Perform a single training step on a batch of data.

    Args:
        model (torch.nn.Module): Model to train.
        xts_batch (torch.Tensor): Batch of noised sequences.
        x0s_batch (torch.Tensor): Batch of original sequences (one-hot encoded).
        timesteps_batch (torch.Tensor): Batch of timesteps.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        use_compute_mse_loss (bool): Whether to use the MSE loss.
        use_cross_entropy_loss (bool): Whether to use the cross-entropy loss.
        rescale_loss (bool): Whether to rescale the loss based on alpha and alpha_bar.
        alpha_bars (torch.Tensor, optional): Schedule of alpha_bar values.
        alphas (torch.Tensor, optional): Schedule of alpha values.
        scaler (torch.amp.GradScaler, optional): Gradient scaler for mixed precision training.
    
    Returns:
        torch.Tensor: Training loss (before mean reduction).
        torch.Tensor: Predicted starting points (detached from the graph).
    """
    with torch.amp.autocast('cuda', enabled=(scaler is not None)):
        # Denoise the batch with the neural network
        x0_hats_batch = model(xts_batch, timesteps_batch)

        if rescale_loss:
            # Compute the loss coefficients for continuous diffusion using the new normalized coefficients to prevent gradient explosion
            rescale_coeffs = get_loss_coefficients_continuous_new(alpha_bars, alphas, timesteps_batch)
        else:
            rescale_coeffs = None
        
        if use_compute_mse_loss:
            training_loss = compute_mse_loss(x0s_batch, x0_hats_batch, rescale_coeffs = rescale_coeffs)
        elif use_cross_entropy_loss:
            # Convert x0s_batch from one-hot to class labels
            x0s_batch = torch.argmax(x0s_batch, dim=-1).long()
            training_loss = compute_cross_entropy_loss(
                x0s_batch, x0_hats_batch, rescale_coeffs = rescale_coeffs
            )
        else:
            raise ValueError("At least one of use_compute_mse_loss or use_cross_entropy_loss must be True.")
        
        training_loss = training_loss.mean()
    
    optimizer.zero_grad(set_to_none=True)

    if scaler is None:                       # plain FP32 path
        training_loss.backward()
        # Gradient clipping to prevent explosion (safety net), disabled to match original controlled setting diffusion training dynamics
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    else:                                    # mixed-precision path
        scaler.scale(training_loss).backward()
        # Unscale before clipping to get true gradient magnitudes, disabled to match original controlled setting diffusion training dynamics
        # scaler.unscale_(optimizer)
        # Gradient clipping to prevent explosion (safety net), disabled to match original controlled setting diffusion training dynamics
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    return training_loss.detach(), x0_hats_batch.detach()


def train_model(
    model,
    data_gen_args,
    optimizer,
    n_epochs,
    batch_size,
    use_compute_mse_loss=False,
    use_cross_entropy_loss=True,
    scheduler=None,
    use_wandb=False,
    checkpointing_period_epochs=None,
    model_dir=None,
    checkpoint_id=None,
    save_final_model=False,
    count_fun=None,
    epoch_start=0,
    reweighting=False,
    rescale_loss=False,
):
    """
    Train a model.

    Args:
        model (torch.nn.Module): Model to train.
        data_gen_args (dict): Arguments for the data generator.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        n_epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        use_compute_mse_loss (bool): Whether to use the MSE loss.
        use_cross_entropy_loss (bool): Whether to use the cross-entropy loss.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        use_wandb (bool): Whether to use Weights & Biases logging.
        checkpointing_period_epochs (int, optional): Number of checkpoints to save.
        model_dir (str, optional): Directory in which to save the model.
        checkpoint_id (str, optional): Identifier for the checkpoint file names.
        save_final_model (bool): Whether to save the final model at the end.
        count_fun (callable, optional): Deprecated. Kept for checkpoint backward compatibility; not used during training.
        epoch_start (int): Starting epoch (useful for resuming training).
        reweighting (bool): Whether to use reweighting in data generation.
        rescale_loss (bool): Whether to rescale the loss based on alpha and alpha_bar.
    Returns:
        torch.nn.Module: The trained model.
        dict: Training history containing losses and learning rates.
    """
    # Retrieve data generator parameters
    x0s = data_gen_args['x0s']
    t_final = data_gen_args['t_final']
    alpha_bars = data_gen_args['alpha_bars']
    alphas = data_gen_args['alphas']
    device = data_gen_args['device']

    # Log scheduler info
    if scheduler is not None:
        print('Using learning rate scheduler:', scheduler)
    else:
        print('No learning rate scheduler provided.')

    # Set up model checkpointing
    if (checkpointing_period_epochs is not None) or save_final_model:
        if not os.path.exists(model_dir):
            print(f'Creating directory for saving model: {model_dir}')
            os.makedirs(model_dir)
        model_params_path = os.path.join(model_dir, 'model_params.json')
        with open(model_params_path, 'w') as f:
            json.dump(model.get_params_dict(), f)

    # Set the checkpoint epochs to save        
    if checkpointing_period_epochs is not None:
        epochs_to_save = torch.logspace(2, torch.log10(torch.tensor(n_epochs)), checkpointing_period_epochs).long().tolist()
        print(f'Epochs to save: {epochs_to_save}')
    else:
        epochs_to_save = []
   
    epoch_counter = epoch_start
    training_history = {
        'training_loss': [],
        'learning_rate': []
    }

    # Initialize diffusion dataset
    dataset = DiffusionDataset(x0s, t_final, alpha_bars,
                            reweight=reweighting,
                            device='cpu')
    
    # Create DataLoader for the dataset
    training_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=os.cpu_count() // 4,   # 4-8 typical
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Use gradient scaling for mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    # Start of training loop
    with trange(epoch_start, n_epochs, desc='Training') as pbar:
        for t_epoch in pbar:
            epoch_counter += 1
            training_loss_batches = []

            for batch in training_loader:
                xts_batch, x0s_batch, timesteps_batch = batch
                xts_batch = xts_batch.to(device, non_blocking=True)
                x0s_batch = x0s_batch.to(device, non_blocking=True)
                timesteps_batch = timesteps_batch.to(device, non_blocking=True)

                # Training step for one batch
                loss_batch, _ = training_step(
                    model=model,
                    xts_batch=xts_batch,
                    x0s_batch=x0s_batch,
                    timesteps_batch=timesteps_batch,
                    optimizer=optimizer,
                    use_compute_mse_loss=use_compute_mse_loss,
                    use_cross_entropy_loss=use_cross_entropy_loss,
                    rescale_loss=rescale_loss,
                    alpha_bars=alpha_bars,
                    alphas=alphas,
                    scaler=scaler,
                )
                training_loss_batches.append(loss_batch.mean())

            # Update learning rate at the end of the epoch
            if scheduler is not None:
                scheduler.step()

            # Save checkpoints
            if checkpointing_period_epochs is not None:
                if epoch_counter in epochs_to_save:
                    checkpoint_path = os.path.join(
                        model_dir, 
                        checkpoint_id + f'_epoch_{epoch_counter}.pt'
                    )
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_history': training_history,
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                            'count_fun': count_fun
                        },
                        checkpoint_path
                    )

            # Compute the average training loss for the epoch
            epoch_loss = torch.tensor(training_loss_batches).mean()
            training_history['training_loss'].append(epoch_loss)
            training_history['learning_rate'].append(
                optimizer.state_dict()['param_groups'][0]['lr']
            )

            # Log metrics to Weights & Biases if enabled
            if use_wandb:
                wandb.log({'learning_rate': optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch_counter)
                wandb.log({'training_loss': epoch_loss}, step=epoch_counter)

            pbar.set_postfix(
                training_loss=training_history['training_loss'][-1],
                learning_rate=training_history['learning_rate'][-1]
            )

    # Convert training loss history to a list of floats
    training_history['training_loss'] = torch.tensor(training_history['training_loss']).tolist()

    # Save the final model checkpoint if required
    if save_final_model:
        final_checkpoint_path = os.path.join(
            model_dir, 
            checkpoint_id + f'_epoch_{epoch_counter}.pt'
        )
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'training_history': training_history,
                'count_fun': count_fun
            },
            final_checkpoint_path
        )

    return model, training_history
