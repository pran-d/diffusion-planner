import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_forcing_transformer.guidance_functions import guidance_smoothness
from diffusion_forcing_transformer.discrete_diffusion import DiscreteDiffusion
from diffusion_forcing_transformer.history_guidance import HistoryGuidance
from diffusion_forcing_transformer.torch_utils import bernoulli_tensor
from typing import Optional, Callable, Tuple, Dict
from functools import partial
from tqdm import tqdm
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np
from utils.configcls import Config

# =========================
# Conditioning Embeddings
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class HistoryAttention(nn.Module):
    def __init__(self, state_dim, history_len):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, state_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=state_dim, num_heads=1, batch_first=True
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x):
        # x: (B, C, H) or (B, C)
        if x.ndim == 2: x = x.unsqueeze(-1) # (B, C, 1)
        x = x.permute(0, 2, 1)  # (B, H, C)
        q = self.query.expand(x.size(0), -1, -1)  # (B, 1, C)

        out, _ = self.attn(q, x, x)  # (B, 1, C)
        return self.norm(out.squeeze(1))  # (B, C)
        

class DFoTTrajectory(nn.Module):
    def __init__(self, model_config, data_config, noise_scheduler_config=None, noise_scheduler=None):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.noise_scheduler_config = noise_scheduler_config 
        self.noise_scheduler = noise_scheduler
        
        self.x_shape = torch.Size([data_config['num_features']])
        self.state_condition = model_config.get("state_condition", False)
        self.task_condition = model_config.get("task_condition", False)
        
        state_dim = model_config['hidden_size'] if self.state_condition else 0
        task_dim = model_config.get("hidden_size", 64) if self.task_condition else 0

        self.external_cond_dim = state_dim + task_dim
            
        if self.state_condition:
            self.state_embedding = MLP(
                in_dim=data_config.get("num_observations", 86),
                out_dim=model_config.get("hidden_size", 64)
            )

        if self.task_condition:
            self.task_embedding = MLP(
                in_dim=data_config.get("num_task_params", 10),
                out_dim=model_config.get("hidden_size", 64) 
            )
  
        # Backbone config
        backbone_type = model_config.get("backbone_type", "dit1d")
        self.backbone_type = backbone_type

        if backbone_type == "dit1d":
            backbone_cfg = Config({
                "name": "dit1d",
                "hidden_size": model_config.get("hidden_size", 256),
                "depth": model_config.get("depth", 12),
                "num_heads": model_config.get("num_heads", 4),
                "mlp_ratio": model_config.get("mlp_ratio", 4.0),
                "pos_emb_type": "learned_1d",
                "use_gradient_checkpointing": False,
                "external_cond_dropout": model_config.get("external_cond_dropout", 0.1),
            })
        elif backbone_type == "unet1d":
            backbone_cfg = Config({
                "name": "unet1d",
                "hidden_size": model_config.get("hidden_size", 64),
                "channel_mult": model_config.get("channel_mult", (1, 2, 4, 8)),
                "num_res_blocks": model_config.get("num_res_blocks", 2),
                "attn_resolutions": model_config.get("attn_resolutions", (2, 3)),
                "external_cond_dropout": model_config.get("external_cond_dropout", 0.1),
            })
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # Map beta schedule
        beta_schedule = self.noise_scheduler_config.get("beta_schedule", "squaredcos_cap_v2")

        # Map prediction type
        prediction_type = self.noise_scheduler_config.get("prediction_type", "v_prediction")
        objective = "pred_v" if prediction_type == "v_prediction" else "pred_noise"

        # Diffusion config
        diffusion_cfg = Config({
            "timesteps": self.noise_scheduler_config.get("train_timesteps", 1000),
            "sampling_timesteps": self.noise_scheduler_config.get("inference_timesteps", 50),
            "beta_schedule": beta_schedule,
            "schedule_fn_kwargs": {"shift": 1.0},
            "objective": objective,
            "loss_weighting": {
                "strategy": "fused_min_snr",
                "snr_clip": 5.0,
                "cum_snr_decay": 0.9,
            },
            "ddim_sampling_eta": 0.0,
            "clip_noise": 20.0,
            "use_causal_mask": self.noise_scheduler_config.get("use_causal_mask", False),
        })
        
        # Extract betas from noise_scheduler if available
        betas = None
        if self.noise_scheduler is not None:
            betas = self.noise_scheduler.betas

        self.max_tokens = data_config['num_timesteps'] // data_config.get('downsample', 1)

        self.diffusion_model = DiscreteDiffusion(
            cfg=diffusion_cfg,
            backbone_cfg=backbone_cfg,
            x_shape=self.x_shape,
            max_tokens=self.max_tokens,
            external_cond_dim=self.external_cond_dim,
            betas=betas,
        )
        
        # Print parameter count
        total_params = sum(p.numel() for p in self.diffusion_model.model.parameters())
        print(f"Initialized {backbone_type} backbone with {total_params/1e6:.2f}M parameters.")
        
        self.timesteps = diffusion_cfg.timesteps
        self.sampling_timesteps = diffusion_cfg.sampling_timesteps
        self.clip_noise = diffusion_cfg.clip_noise
        self.use_causal_mask = diffusion_cfg.use_causal_mask
        
        self.noise_level = self.noise_scheduler_config.get("noise_level", "random_independent")
        self.scheduling_matrix = self.noise_scheduler_config.get("scheduling_matrix", "full_sequence")
        self.uniform_future = False

        self.is_full_sequence = (
            self.noise_level == "random_uniform"
        )
        
        self.generator = None # Can be set externally if needed



    #### Training Utils ####
    def _get_training_noise_levels(
            self, xs: torch.Tensor, masks: torch.Tensor = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Generate random noise levels for training."""
            batch_size, n_tokens, *_ = xs.shape

            # random function different for continuous and discrete diffusion
            rand_fn = partial(
                *(
                    (torch.randint, 0, self.timesteps)
                ),
                device=xs.device,
                generator=self.generator,
            )

            match self.noise_level:
                case "random_independent":  # independent noise levels (Diffusion Forcing)
                    noise_levels = rand_fn((batch_size, n_tokens))
                case "random_uniform":  # uniform noise levels (Typical Video Diffusion)
                    noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)

            if self.uniform_future:  # simplified training (Appendix A.5)
                noise_levels = rand_fn((batch_size, 1)).repeat(
                    1, n_tokens
                )

            # treat frames that are not available as "full noise"
            if masks is not None:
                noise_levels = torch.where(
                    reduce(masks.bool(), "b t ... -> b t", torch.any),
                    noise_levels,
                    torch.full_like(
                        noise_levels,
                        self.timesteps - 1,
                    ),
            )
                
            return noise_levels, masks
    
    #### ============== ####

    def _reweight_loss(self, loss, weight=None):
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss

    def forward(self, x, model_cond=None, masks=None, timesteps=None):
        """
        Forward pass for training.
        x: (B, C, T) - RobotDiffuser passes (B, C, T)
        init_cond: (B, C, H) or similar - External condition
        masks: (B, T) - Boolean mask indicating valid tokens
        timesteps: (B, T) - Per-token timesteps (noise levels)
        """
        # Create default mask (all valid initially)
        if masks is None:
            masks = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        
        state_cond = None
        task_cond = None

        if model_cond is not None:
            if isinstance(model_cond, (list, tuple)):
                idx = 0
                if self.state_condition:
                    if len(model_cond) > idx: state_cond = model_cond[idx]
                    idx += 1
                if self.task_condition:
                    if len(model_cond) > idx: task_cond = model_cond[idx]
                    idx += 1

            else:
                # Single condition provided
                if self.state_condition:
                    state_cond = model_cond
                elif self.task_condition:
                    task_cond = model_cond
        
        cond_list = []
        if self.state_condition and state_cond is not None:
            s_cond = self.state_embedding(state_cond) 
            if s_cond.ndim == 2:
                # (B, 1, D)
                s_cond = s_cond.unsqueeze(1)
            cond_list.append(s_cond)
        if self.task_condition and task_cond is not None:
            t_cond = self.task_embedding(task_cond)
            if t_cond.ndim == 2:
                t_cond = t_cond.unsqueeze(1)
            cond_list.append(t_cond)

        ext_cond = None
        if cond_list:
            # Determine the maximum time dimension
            max_t = max(c.shape[1] for c in cond_list)

            # Broadcast each condition to match max_t
            cond_list_broadcast = [
                c.repeat(1, max_t, 1) if c.shape[1] == 1 else c
                for c in cond_list
            ]

            # Concatenate along feature dimension
            ext_cond = torch.cat(cond_list_broadcast, dim=-1)


        # Generate noise levels if not provided
        if timesteps is None:
            # Generate noise levels
            noise_levels, masks = self._get_training_noise_levels(x, masks)
            k = noise_levels
        else:
            k = timesteps
        
        model_pred, model_out, loss = self.diffusion_model(x, ext_cond, k)
        
        # Reweight loss using masks
        loss = self._reweight_loss(loss, masks)
        
        output_dict = {
            "loss": loss,
            "xs_pred": model_pred,
            "xs": model_out
        }
        return output_dict

    def sample(self, num_trajectories, model_cond=None, cfg_w=0.0):
        """
        Sampling method.
        """
        state_cond_input = None
        task_cond = None

        if isinstance(model_cond, (list, tuple)):
            # Flatten list if nested
            flat_cond = []
            for item in model_cond:
                if isinstance(item, (list, tuple)) and len(item) > 0 and not isinstance(item[0], str):
                    flat_cond.extend(item)
                else:
                    flat_cond.append(item)

            idx = 0
            if len(flat_cond) > idx and self.state_condition: 
                state_cond_input = flat_cond[idx]
                idx += 1
            if len(flat_cond) > idx and self.task_condition:
                task_cond = flat_cond[idx]
                idx += 1
        else:
            if self.state_condition:
                state_cond_input = model_cond
            elif self.task_condition:
                task_cond = model_cond

        cond_list = []
        if self.state_condition and state_cond_input is not None:
            s_cond = self.state_embedding(state_cond_input) # (B, C)
            cond_list.append(s_cond)
        if self.task_condition and task_cond is not None:
            t_cond = self.task_embedding(task_cond) # (B, D)
            cond_list.append(t_cond)
        
        # If we have external conditions (init_cond), pass them as conditions
        conditions = None
        if cond_list:
            # Determine the maximum time dimension
            max_t = max(c.shape[1] if c.ndim == 3 else 1 for c in cond_list)

            # Broadcast each condition to match max_t
            cond_list_broadcast = []
            for c in cond_list:
                if c.ndim == 2:  # add time dimension if missing
                    c = c.unsqueeze(1)
                if c.shape[1] == 1 and max_t > 1:  # broadcast along time
                    c = c.repeat(1, max_t, 1)
                cond_list_broadcast.append(c)

            # Concatenate all conditions along feature dimension
            conditions = torch.cat(cond_list_broadcast, dim=-1)

        xs_pred, _ = self._sample_sequence(
            batch_size=num_trajectories,
            conditions=conditions, # External conditions
            cfg_w=cfg_w,
        )
        
        # Return (B, T, C)
        return xs_pred

    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Extend the tensor by adding dimensions at the end to match x_stacked_shape."""
        return rearrange(x, "... -> ..." + " 1" * len(self.x_shape))

    def _generate_pyramid_scheduling_matrix(self, horizon, sampling_timesteps):
        K = sampling_timesteps
        total_steps = horizon * K
        
        # j goes from 0 to total_steps
        j = np.arange(total_steps + 1).reshape(-1, 1) # (Total+1, 1)
        t = np.arange(horizon).reshape(1, -1)         # (1, Horizon)
        
        # We want noise level n(t, j)
        # n(t, j) = K - (j - t)
        # Clamped to [0, K]
        # If j < t, n = K (Full noise)
        # If j > t + K, n = 0 (Clean)
        
        noise_level = K - (j - t)
        noise_level = np.clip(noise_level, 0, K)
        
        return noise_level

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        padding: int = 0,
    ):
        if self.scheduling_matrix == "full_sequence":
            scheduling_matrix = np.arange(self.sampling_timesteps, -1, -1)[
                :, None
            ].repeat(horizon, axis=1)
        elif self.scheduling_matrix == "autoregressive":
             scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                 horizon, self.sampling_timesteps
             )
        else:
             raise ValueError(f"Unknown scheduling matrix type: {self.scheduling_matrix}")
        
        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

        scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(
            scheduling_matrix
        )

        # paded entries are labeled as pure noise
        scheduling_matrix = F.pad(
            scheduling_matrix, (0, padding, 0, 0), value=self.timesteps - 1
        )

        return scheduling_matrix

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        guidance_wt: float = 1.0,
        history_guidance: Optional[HistoryGuidance] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
        cfg_w: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        x_shape = self.x_shape

        if length is None:
            length = self.max_tokens

        horizon = length 
        padding = horizon - length
        
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.diffusion_model.parameters().__next__().device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        # create empty context and zero context mask
        context_mask = torch.zeros(
            (batch_size, horizon), dtype=torch.long, device=xs_pred.device
        )

        if history_guidance is None:
            # by default, use conditional sampling
            history_guidance = HistoryGuidance.conditional(
                timesteps=self.timesteps,
                visualize=False
            )

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            horizon - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(xs_pred.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
    
        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            with history_guidance(context_mask) as history_guidance_manager:
                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)
                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
                    history_guidance_manager.prepare(
                        xs_pred,
                        from_noise_levels,
                        to_noise_levels,
                        replacement_fn=self.diffusion_model.q_sample,
                        replacement_only=self.is_full_sequence,
                    )
                )

                # update xs_pred by DDIM or DDPM sampling
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    (
                        repeat(
                            conditions,
                            "b ... -> (b nfe) ...",
                            nfe=nfe,
                        ).clone()
                        if conditions is not None
                        else None
                    ),
                    conditions_mask,
                    guidance_fn=guidance_fn,
                    guidance_wt=guidance_wt,
                    cfg_w=cfg_w,
                )

                xs_pred = history_guidance_manager.compose(xs_pred)
            
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record
