from matplotlib.pylab import sample
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from matplotlib import pyplot as plt
import os
import yaml
import mujoco

from diffusers import EMAModel
from datasets.conditional_dataset import ConditionalStateDataset
from models.model import RobotDiffuser
from config.configure import load_config, get_data_path, get_save_path, get_log_path, get_norm_path

def apply_condition_dropout(
    cond,
    dropout_prob: float,
):
    """
    Apply per-sample dropout for classifier-free guidance.

    Args:
        cond: torch.Tensor or None, shape (B, ...)
        dropout_prob: Probability of dropping condition
        return_mask: If True, also return the keep mask

    Returns:
        dropped_cond or (dropped_cond, keep_mask)
    """
    if cond is None or dropout_prob <= 0.0:
        return cond

    if not torch.is_tensor(cond):
        cond = torch.as_tensor(cond)

    bs = cond.shape[0]
    device = cond.device

    drop_mask = torch.bernoulli(
        torch.full((bs,), dropout_prob, device=device)
    )

    # Broadcast mask across feature dimensions
    view_shape = (bs,) + (1,) * (cond.ndim - 1)
    drop_mask_broadcast = drop_mask.view(view_shape)

    return torch.where(
        drop_mask_broadcast.bool(),
        torch.rand_like(cond),
        cond
    )

def apply_state_condition_noise(
    state,
    noise_cfg: dict,
):
    """
    Apply per-feature Gaussian noise to state conditioning.

    Args:
        state: torch.Tensor or None, shape (B, F) or (B, T, F)
        noise_cfg: Dict with entries:
            {name: {"start": int, "end": int, "level": float}}

    Returns:
        noised_state or (noised_state, noise_added)
    """
    if state is None or not noise_cfg:
        return state

    if not torch.is_tensor(state):
        state = torch.as_tensor(state)

    device = state.device
    noised = state.clone()

    feat_dim = noised.ndim - 1

    for name, cfg in noise_cfg.items():
        start = cfg["start"]
        end = cfg["end"]
        level = cfg["level"]

        if level <= 0.0:
            continue

        slicer = [slice(None)] * noised.ndim
        slicer[feat_dim] = slice(start, end)
        slicer = tuple(slicer)

        noise = torch.randn(
            *noised[slicer].shape,
            device=device
        ) * level

        noised[slicer] += noise

    return noised

def apply_trajectory_swapping(
    state,
    swap_prob: float,
    start: int = 0,
    end: int = -1,
    alpha: float = 1.0,
):
    """
    Apply goal swapping to the goal portion of the state conditioning.

    Args:
        state: torch.Tensor or None, shape (B, F)
        swap_prob: Probability of swapping goal with random noise
        return_swapped_mask: If True, also return the swapped mask

    Returns:
        swapped_state 
    """
    if state is None or swap_prob <= 0.0:
        return state

    if not torch.is_tensor(state):
        state = torch.as_tensor(state)

    bs = state.shape[0]
    device = state.device

    swapped_mask = torch.bernoulli(
        torch.full((bs,), swap_prob, device=device)
    ).bool()
        
    perm = torch.randperm(state.shape[0])
    state_swapped = state[perm]

    state[..., start:end] = torch.where(
        swapped_mask.unsqueeze(-1),
        (1 - alpha) * state_swapped[..., start:end] + alpha * state[..., start:end],
        state[..., start:end]
    )

    return state


# ===============================
# Setup
# ===============================
with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

model_cfg, data_cfg, training_cfg, noise_schedule_cfg = load_config("config/config.yaml", config.get("auto_conf", False))

save_dir = get_save_path(model_cfg, data_cfg, training_cfg)
os.makedirs(save_dir, exist_ok=True)

# Preserve resume parameters from the current config
if config.get("resume", False):        
    resume_checkpoint = config.get("resume_checkpoint", 0)
else:
    saved_config_path = os.path.join(save_dir, "config.yaml")
    # Save the current configuration
    print(f"Starting new training: Saving configuration to {saved_config_path}...\n")
    full_config = {
        "model": model_cfg,
        "data": data_cfg,
        "training": training_cfg,
        "noise_scheduler": noise_schedule_cfg
    }
    with open(saved_config_path, 'w') as f:
        yaml.dump(full_config, f)

state_condition = model_cfg.get("state_condition", False)
history_condition = model_cfg.get("history_condition", False)
text_condition = model_cfg.get("text_condition", False)
action_condition = model_cfg.get("action_condition", False)
goal_condition = model_cfg.get("goal_condition", False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diffuser = RobotDiffuser(
    model_config=model_cfg,
    data_config=data_cfg,
    training_config=training_cfg,
    noise_scheduler_config=noise_schedule_cfg,
    mode='train',
    device=device
)

optimizer = torch.optim.AdamW(
    diffuser.model.parameters(), 
    lr=training_cfg.get("learning_rate", 1e-4),
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=training_cfg["num_epochs"], \
    eta_min=1e-6
)

scaler = GradScaler()

ema = EMAModel(
    diffuser.model.parameters(),
    decay=0.9995,
    update_after_step=0,
)

starting_epoch = 0
if config.get("resume", False):
    starting_epoch = resume_checkpoint
    checkpoint = diffuser.loadWeights(starting_epoch)
    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

data_path = get_data_path(data_cfg)
norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)
data = {}

dataset = ConditionalStateDataset(
    dataset_path=data_path, config=data_cfg, state_condition=state_condition, history_condition=history_condition, 
    goal_condition=goal_condition, action_condition=action_condition, load_norm=True, norm_path=norm_path
)

use_reconstruction_loss = ("x_original" in data) and training_cfg.get("use_reconstruction_loss", False) and data_cfg.get("predict")=="qdiff"

train_dataloader = DataLoader(
    dataset, 
    batch_size=data_cfg["batch_size"], 
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# Load MuJoCo
xml_path = "./mj_model.xml"
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)

# Load norm stats
if dataset.norm_path and os.path.exists(dataset.norm_path):
    print(f"Getting norm constants from {dataset.norm_path}...\n")
    norm_stats = np.load(dataset.norm_path)

    if "min_action" in norm_stats:
        min_action = torch.tensor(norm_stats["min_action"], device=device)
        max_action = torch.tensor(norm_stats["max_action"], device=device)

    if "min_history" in norm_stats:
        min_history = torch.tensor(norm_stats["min_history"], device=device)
        max_history = torch.tensor(norm_stats["max_history"], device=device)

    if "min_current_state" in norm_stats:
        min_current_state = torch.tensor(norm_stats["min_current_state"], device=device)
        max_current_state = torch.tensor(norm_stats["max_current_state"], device=device)

    if "min_future" in norm_stats:
        min_future = torch.tensor(norm_stats["min_future"], device=device)
        max_future = torch.tensor(norm_stats["max_future"], device=device)
else:
    stats_p = os.path.join(real_path, "stats.npz")
    if os.path.exists(stats_p):
        print(f"Found pre-computed stats at {stats_p}...\n")
        norm_stats = np.load(norm_path)
        np.savez(norm_path, **norm_stats)


# ===============================
# TensorBoard Setup
# ===============================
losses = []
epoch_losses_history = []
window_size = 100

log_dir = get_log_path(model_cfg, data_cfg, training_cfg)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# ===============================
# Training Loop
# ===============================
diffuser.model.train()

num_epochs = training_cfg["num_epochs"] + 1
for epoch in range(starting_epoch, num_epochs):
    epoch_losses = []
    for step, batch in enumerate(train_dataloader):
        
        if training_cfg.get("batches_per_epoch") and step >= training_cfg["batches_per_epoch"]:
            break

        state_cond = None
        history_cond = None
        text_input = None
        original_states = None
        goal_input = None    

        # Unpack batch
        batch_data = list(batch)

        prediction_target = batch_data[0].to(device)
        
        idx = 1
        if state_condition:
            state_cond = batch_data[idx].to(device)

            # goal swapping
            state_cond = apply_trajectory_swapping(
                state_cond,
                training_cfg.get("swapping_probabilities", {})["goal"],
                start=86,
                end=89,
                alpha=torch.rand(1).uniform_(0.4, 1.0).item()
            )
            # joint position swapping
            state_cond = apply_trajectory_swapping(
                state_cond,
                training_cfg.get("swapping_probabilities", {})["joint_pos"],
                start=0,
                end=86,
                alpha=torch.rand(1).uniform_(0.4, 1.0).item()
            )
            state_cond = apply_state_condition_noise(
                state_cond,
                training_cfg.get("state_conditioning_noise_level", {}),
            )
            state_cond = apply_condition_dropout(
                state_cond, 
                training_cfg.get("condition_dropout_prob", 0.0)
            )
            idx += 1
        
        if goal_condition:
            goal_input = batch_data[idx].to(device)

            goal_input = apply_condition_dropout(
                goal_input, 
                training_cfg.get("condition_dropout_prob", 0.0)
            )
            idx += 1

        if history_condition:
            history_cond = batch_data[idx].to(device)
            history_cond = apply_condition_dropout(
                history_cond, 
                training_cfg.get("condition_dropout_prob", 0.0)
            )
            idx += 1
            
        if text_condition:
            text_input = batch_data[idx].to(device)
            text_input = apply_condition_dropout(
                text_input, 
                training_cfg.get("condition_dropout_prob", 0.0)
            )
            idx += 1
            
        # Construct cond for model
        cond = []
        if state_cond is not None: cond.append(state_cond)
        if history_cond is not None: cond.append(history_cond)
        if goal_input is not None: cond.append(goal_input)
        if text_input is not None: cond.append(text_input)
    
        model_cond = tuple(cond) if len(cond) > 0 else None
        if len(cond) == 1: model_cond = cond[0]
        
        bs, _, ts = prediction_target.shape

        if model_cfg["type"] == "dfot":
            # timesteps = torch.randint(
            #     0, diffuser.noise_scheduler.config.num_train_timesteps, (bs, ts,), device=device
            # ).long()
            timesteps = None
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                diff_output = diffuser.model(
                    prediction_target, 
                    model_cond, 
                    timesteps=timesteps, 
                )
                model_pred = diff_output["xs_pred"]
                pred_loss = diff_output["loss"]
                loss = pred_loss.mean()

        else:
            # Standard Diffusion: Per-sample timesteps
            # Sample random noise
            noise = torch.randn_like(prediction_target)  

            # Sample random timesteps
            timesteps = torch.randint(
                0, diffuser.noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()

            # mixed precision 32
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                    noisy_traj = diffuser.noise_scheduler.add_noise(prediction_target, noise, timesteps)

                    # Predict the noise added for t timesteps (equivalent to predicting original sample) using UNet
                    model_pred = diffuser.model(noisy_traj, timesteps, model_cond).sample

                    # Define the target for the network (velocity or noise)
                    if noise_schedule_cfg["prediction_type"]=="v_prediction":
                        target = diffuser.noise_scheduler.get_velocity(prediction_target, noise, timesteps)
                    elif noise_schedule_cfg["prediction_type"]=="epsilon":
                        target = noise
                    elif noise_schedule_cfg["prediction_type"]=="sample":
                        target = prediction_target
                    # Compute MSE loss
                    loss = F.mse_loss(model_pred, target, reduction="mean")

        optimizer.zero_grad()
        
        # scale the loss function and compute gradients
        scaler.scale(loss).backward()
        
        # unscale gradients and clips 
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(diffuser.model.parameters(), 1.0)
        
        # optimizer step is called if gradients do not contain infs or NaNs
        scaler.step(optimizer)
        
        # updates the scale for next iteration
        scaler.update()

        # use ema to update the model average
        # ema.step(diffuser.model.parameters())
        
        losses.append(loss.item())
        epoch_losses.append(loss.item())

        # Step-wise logging
        global_step = epoch * len(train_dataloader) + step
        writer.add_scalar("Loss/train_step", loss.item(), global_step)
        writer.add_scalar("Loss/pred_loss", pred_loss.mean().item(), global_step)


    # Epoch summary
    mean_loss = np.mean(epoch_losses)
    epoch_losses_history.append(mean_loss)
    writer.add_scalar("Loss/train_epoch", mean_loss, epoch)
    
    scheduler.step()
    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

    window_mean = np.mean(epoch_losses_history[-window_size:])
    print(f"Epoch [{epoch}/{num_epochs}] - Mean Loss: {window_mean:.5f}")
    if epoch % training_cfg.get("save_every", 50) == 0:
        checkpoint = {
            "model": diffuser.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict()
        }

        torch.save(checkpoint, f"{save_dir}/model_{epoch}.pth")
    
    # if epoch % 500 == 0:
    #     diffuser.save_ema_weights(ema.shadow_params, f"{save_dir}/ema_model_{epoch}.pth")

# ===============================
# Plot Loss Curve (after training)
# ===============================
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(losses)
axs[0].set_title("Loss over iterations")

axs[1].plot(np.log(losses))
axs[1].set_title("Log Loss")

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

writer.close()
