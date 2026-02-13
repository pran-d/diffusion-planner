from matplotlib.pylab import cond, sample
import torch
import numpy as np
from torch import cond, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from matplotlib import pyplot as plt
import os
import yaml
import mujoco

from diffusers import EMAModel
from datasets import FlexibleWindowDataset, ConditionalStateDataset
from models.model import RobotDiffuser
from config.configure import load_config, get_data_path, get_save_path, get_log_path, get_norm_path

# ===============================
# Setup
# ===============================
with open("config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

model_cfg, data_cfg, training_cfg, noise_schedule_cfg = load_config("config/config.yaml", config.get("auto_conf", False))

save_dir = get_save_path(model_cfg, data_cfg, training_cfg)
os.makedirs(save_dir, exist_ok=True)

data_path = get_data_path(data_cfg)
norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)

# Preserve resume parameters from the current config
calculate_stats = True
if config.get("resume", False):        
    resume_checkpoint = config.get("resume_checkpoint", 0)
    if norm_path and os.path.exists(norm_path):
        print(f"Found existing normalization stats at {norm_path}, loading...")
        calculate_stats=False
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
task_condition = model_cfg.get("task_condition", False)

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

if data_cfg.get("dataset_class", "flexible") == "conditional":
    dataset = ConditionalStateDataset(
        dataset_path=data_path, config=data_cfg, 
        state_condition=state_condition, history_condition=False, 
        task_condition=task_condition, action_condition=False, 
        load_norm=True, norm_path=norm_path
    )
else:
    dataset = FlexibleWindowDataset(
        data_root=data_path, config=data_cfg, 
        calculate_stats=calculate_stats, norm_path=norm_path,
        noise_cfg=training_cfg.get("state_conditioning_noise_level", {}),
        add_noise=training_cfg.get("add_obs_noise", False), add_goal_noise=training_cfg.get("add_goal_noise", False)
    )

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

        if batch[0].shape[0] < data_cfg["batch_size"]:
            # Pad the batch by repeating elements if it's smaller than batch_size
            current_bs = batch[0].shape[0]
            target_bs = data_cfg["batch_size"]
            repeats = (target_bs + current_bs - 1) // current_bs
            
            new_batch = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    dims = [1] * item.dim()
                    dims[0] = repeats
                    new_batch.append(item.repeat(*dims)[:target_bs])
            else:
                    new_batch.append(item)
            batch = new_batch

        state_cond = None
        task_cond = None    

        # Unpack batch
        batch_data = list(batch)

        prediction_target = batch_data[0].to(device)
        
        idx = 1
        if state_condition:
            state_cond = batch_data[idx].to(device)
            idx += 1
        
        if task_condition:
            task_cond = batch_data[idx].to(device)
            idx += 1
            
        # Construct cond for model
        cond = []
        if state_cond is not None: cond.append(state_cond)
        if task_cond is not None: cond.append(task_cond)
    
        model_cond = tuple(cond) if len(cond) > 0 else None
        if len(cond) == 1: model_cond = cond[0]
        
        bs, ts, _ = prediction_target.shape

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
    print(f"Epoch [{epoch}/{num_epochs-1}] - Mean Loss: {window_mean:.5f}")
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
