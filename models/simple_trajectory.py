import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.configcls import Config
from diffusers import DDPMScheduler, DDIMScheduler

# Import the existing Backbone
from diffusion_forcing_transformer.dit1d import DiT1D
from models.unet1d import UNet1D

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimpleTrajectoryDiffuser(nn.Module):
    """
    Simplified Diffusion model for trajectory generation.
    Replaces the complex DiscreteDiffusion/DFoT pipeline with a standard DDPM implementation.
    Maintains exact same network structure (Embeddings + DiT1D/UNet1D).
    """
    def __init__(self, model_config, data_config, noise_scheduler_config, noise_scheduler=None, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.noise_scheduler_config = noise_scheduler_config
        
        # --- 1. Dimensions & Configs ---
        # Input shape: (C,) - number of channels per timestep
        self.x_shape = torch.Size((data_config["num_features"],)) 
        self.max_tokens = data_config['num_timesteps'] // data_config.get('downsample', 1)
        
        self.hidden_size = model_config.get("hidden_size", 256)
        
        # --- 2. Conditions ---
        self.state_condition = model_config.get("state_condition", False)
        self.history_condition = model_config.get("history_condition", False)
        self.goal_condition = model_config.get("goal_condition", False)
        self.expand_task_condition = model_config.get("expand_task_condition", False)
        
        self.external_cond_dim = 0
        if self.state_condition: 
            self.external_cond_dim += self.hidden_size
        if self.history_condition: 
            self.external_cond_dim += self.hidden_size
        if self.goal_condition: 
            self.external_cond_dim += self.hidden_size
        if self.expand_task_condition:
            self.external_cond_dim += self.hidden_size
            
        # --- 3. Embedding Networks ---
        self._init_embeddings(data_config, model_config)
        
        # --- 4. Backbone ---
        backbone_type = model_config.get("backbone_type", "dit1d")
        self.backbone_type = backbone_type

        # Construct cfg object expected by DiT1D
        if backbone_type == "dit1d":
            backbone_cfg = Config({
                "name": "dit1d",
                "hidden_size": self.hidden_size,
                "depth": model_config.get("depth", 12),
                "num_heads": model_config.get("num_heads", 4),
                "mlp_ratio": model_config.get("mlp_ratio", 4.0),
                "pos_emb_type": "learned_1d",
                "use_gradient_checkpointing": False,
                # These were part of the previous complex config, kept for compatibility if needed
                "external_cond_dropout": model_config.get("external_cond_dropout", 0.1),
                "use_fourier_noise_embedding": False, 
            })
            model_cls = DiT1D
        elif backbone_type == "unet1d":
            backbone_cfg = Config({
                "name": "unet1d",
                "hidden_size": model_config.get("hidden_size", 64),
                "channel_mult": model_config.get("channel_mult", (1, 2, 4, 8)),
                "num_res_blocks": model_config.get("num_res_blocks", 2),
                "attn_resolutions": model_config.get("attn_resolutions", (2, 3)),
                "external_cond_dropout": model_config.get("external_cond_dropout", 0.1),
                 "use_fourier_noise_embedding": False,
            })
            model_cls = UNet1D
        else:
             raise ValueError(f"Unknown backbone: {backbone_type}")

        self.model = model_cls(
            cfg=backbone_cfg,
            x_shape=self.x_shape,
            max_tokens=self.max_tokens,
            external_cond_dim=self.external_cond_dim,
            use_causal_mask=False, # Standard diffusion uses full mask usually
        )
        
        # --- 5. Noise Scheduler ---
        # Use the passed scheduler or create a new one (consistent with train.py)
        self.noise_scheduler = noise_scheduler
        if self.noise_scheduler is None:
             self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=noise_scheduler_config.get("train_timesteps", 1000),
                beta_schedule=noise_scheduler_config.get("beta_schedule", "squaredcos_cap_v2"),
                prediction_type=noise_scheduler_config.get("prediction_type", "v_prediction"),
             )
        
        print(f"Initialized Simplified Diffuser ({backbone_type}) | Cond Dim: {self.external_cond_dim}")

    def _init_embeddings(self, data_cfg, model_cfg):
        if self.state_condition:
            obs_dim = data_cfg.get("num_observations", 45)
            self.state_embedding = MLP(obs_dim, self.hidden_size)
        if self.history_condition:
            hist_dim = data_cfg.get("num_features", 48) # Approx
            self.history_embedding = MLP(hist_dim, self.hidden_size)
        if self.goal_condition:
            goal_dim = data_cfg.get("num_features", 48)
            self.goal_embedding = MLP(goal_dim, self.hidden_size)
        if self.expand_task_condition:
            self.task_embedding = MLP(1, self.hidden_size)

    def _process_conditions(self, model_cond, device, batch_size):
        """Processes internal condition logic to produce external_cond tensor."""
        state_cond, history_cond, goal_cond = None, None, None
        
        # Unpack tuple/list if necessary
        if model_cond is not None:
            if isinstance(model_cond, (list, tuple)):
                idx = 0
                if self.state_condition:
                    if len(model_cond) > idx: state_cond = model_cond[idx]
                    idx += 1
                if self.history_condition:
                    if len(model_cond) > idx: history_cond = model_cond[idx]
                    idx += 1
                if self.goal_condition:
                    if len(model_cond) > idx: goal_cond = model_cond[idx]
                    idx += 1
            else:
                 # Assumption: Single condition usually maps to state or goal depending on config
                 if self.state_condition: state_cond = model_cond
                 elif self.goal_condition: goal_cond = model_cond

        cond_list = []
        if self.state_condition and state_cond is not None:
            c = self.state_embedding(state_cond)
            if c.ndim == 2: c = c.unsqueeze(1)
            cond_list.append(c)
            
        if self.history_condition and history_cond is not None:
            c = self.history_embedding(history_cond)
            if c.ndim == 2: c = c.unsqueeze(1)
            cond_list.append(c)
            
        if self.goal_condition and goal_cond is not None:
            c = self.goal_embedding(goal_cond)
            if c.ndim == 2: c = c.unsqueeze(1)
            cond_list.append(c)
            
        if self.expand_task_condition and state_cond is not None:
            # Assuming task is last dim of state
            c = self.task_embedding(state_cond[..., -1:]) 
            if c.ndim == 2: c = c.unsqueeze(1)
            cond_list.append(c)

        if not cond_list:
            return None
        
        # Concatenate: (B, 1, Total_Cond_Dim) or (B, T, Total_Cond_Dim)
        # Broadcast time dimension if needed
        max_t = max([c.shape[1] for c in cond_list])
        cond_list_broadcast = [
            c.repeat(1, max_t, 1) if c.shape[1] == 1 and max_t > 1 else c
            for c in cond_list
        ]
        
        external_cond = torch.cat(cond_list_broadcast, dim=-1)
        return external_cond

    def forward(self, x, model_cond=None, masks=None, timesteps=None):
        """
        Training Forward Pass.
        Returns loss info used by train.py
        """
        # 1. Shape Alignment
        # DFoT (DiT1D) expects (B, T, C). 
        # UNet1D expects (B, C, T).
        
        # Check input shape
        if x.shape[1] == self.x_shape[0]: # Input is (B, C, T)
             if self.backbone_type == "dit1d":
                 x = x.permute(0, 2, 1) # -> (B, T, C)
        else: # Input is (B, T, C)
             if self.backbone_type == "unet1d":
                 x = x.permute(0, 2, 1) # -> (B, C, T)

        # 2. Conditions
        external_cond = self._process_conditions(model_cond, x.device, x.shape[0])
        
        # 3. Noise & Timesteps
        # train.py generates timesteps as (B, T) or (B,) or uses existing if passed
        noise = torch.randn_like(x)
        
        if timesteps is None:
             timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device).long()
        
        # Standardize timesteps to (B,)
        if timesteps.ndim > 1:
            t_sched = timesteps[:, 0]
        else:
            t_sched = timesteps
            
        # Add Noise
        noisy_x = self.noise_scheduler.add_noise(x, noise, t_sched)

        # 4. Model Prediction
        # DiT1D logic for timesteps input
        if self.backbone_type == "dit1d":
             # DiT1D expects (B, T)
             t_input = t_sched.unsqueeze(1).repeat(1, x.shape[1]) # (B, T)
             model_output = self.model(noisy_x, t_input, external_cond=external_cond)
        else:
             # UNet1D expects (B,)
             model_output = self.model(noisy_x, t_sched, cond=external_cond)

        # 5. Loss Calculation
        # Resolve target based on prediction_type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x, noise, t_sched)
        elif self.noise_scheduler.config.prediction_type == "sample":
             target = x
        else:
            raise ValueError(f"Unknown prediction_type {self.noise_scheduler.config.prediction_type}")

        # Compute MSE Loss
        loss = F.mse_loss(model_output, target, reduction="none")
        
        # Masking if provided (e.g., padded sequences)
        if masks is not None:
             # Masks usually (B, T)
             if masks.ndim == 2:
                 masks = masks.unsqueeze(-1) # (B, T, 1)
             
             if self.backbone_type == "unet1d":
                 # UNet output (B, C, T), mask (B, 1, T)
                 masks = masks.permute(0, 2, 1)
                 
             loss = loss * masks
             loss = loss.sum() / masks.sum()
        else:
             loss = loss.mean()
        
        return {
            "loss": loss,
            "xs_pred": model_output, # Return raw output for debugging
            "xs": x
        }

    def sample(self, num_trajectories, model_cond=None, mask_goal=False, cfg_w=1.0, scheduler=None):
        """
        Inference Sampling (DDPM/DDIM).
        Returns: (B, T, C)
        """
        device = self.model.parameters().__next__().device
        
        # 1. Verify scheduler
        if scheduler is None:
            scheduler = self.noise_scheduler
        
        # 2. Conditions
        external_cond = self._process_conditions(model_cond, device, num_trajectories)
        
        # 3. Init Noise
        # Shape: (B, T, C) for DiT, (B, C, T) for UNet
        if self.backbone_type == "dit1d":
             shape = (num_trajectories, self.max_tokens, self.x_shape[0])
        else:
             shape = (num_trajectories, self.x_shape[0], self.max_tokens)
             
        latents = torch.randn(shape, device=device)
        
        # 4. CFG Condition Prep
        if cfg_w != 1.0:
            # We will perform guidance during the loop
            pass # Logic is inside loop
            
        # 5. Denoising Loop
        # self.noise_scheduler timesteps are already set by getSample in model.py usually
        # but let's be safe
        # scheduler.set_timesteps() should have been called externally or we trust it exists
        timesteps = scheduler.timesteps 
        
        for t in tqdm(timesteps, desc="Sampling"):
             # Expand timesteps for batch
             t_batch = torch.full((num_trajectories,), t, device=device, dtype=torch.long)
             
             # Expand for backbone if needed (B, T)
             if self.backbone_type == "dit1d":
                 t_input = t_batch.unsqueeze(1).repeat(1, self.max_tokens)
             else:
                 t_input = t_batch
                 
             # CFG Logic
             if cfg_w != 1.0:
                 # Standard CFG: eps = eps_uncond + w * (eps_cond - eps_uncond)
                 # eps_mod = eps_uncond + (1+w)(eps_cond - eps_uncond) if we reformulate?
                 # Standard: pred = uncond + w * (cond - uncond).
                 # If w=0, pred = cond (wait, no).
                 # Standard Formula: pred = cond + w * (cond - uncond).
                 # If w=0, pred=cond.
                 
                 # Let's use:
                 # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                 # where guidance_scale is (1 + w) if w is the "extra" weight?
                 # Usually users specify cfg_w >= 1.0 (1=no guidance).
                 # But if user says "cfg_w=0.5", they might mean strength.
                 # Let's assume w is the generic multiplier. 
                 # User code `stitch.py` passes `cfg_w`.
                 
                 # Cond Forward
                 noise_pred_cond = self.model(latents, t_input, external_cond=external_cond)
                 
                 # Uncond Forward (Null condition)
                 # For now, zero out the condition. 
                 # Ideally we should use a fixed null embedding learned during training.
                 # Assuming dropout during training makes zeros acceptable.
                 external_cond_uncond = external_cond.clone()
                 external_cond_uncond[:, 128:] = 0.0 # Keep state cond if present, zero out others (e.g., goal)
                 noise_pred_uncond = self.model(latents, t_input, external_cond=external_cond_uncond)
                 
                 noise_pred = noise_pred_uncond + cfg_w * (noise_pred_cond - noise_pred_uncond)

             else:
                 noise_pred = self.model(latents, t_input, external_cond=external_cond)
             
             # Step
             # Scheduler step expects (B, C, T) or (B, T, C) matching sample shape
             step_output = scheduler.step(noise_pred, t, latents)
             latents = step_output.prev_sample

        # 6. Final Shape Check
        if self.backbone_type == "unet1d":
            latents = latents.permute(0, 2, 1) # -> (B, T, C)
            
        return latents
