import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_forcing_transformer.guidance_functions import guidance_goal_mse, guidance_smoothness
from diffusion_forcing_transformer.discrete_diffusion import DiscreteDiffusion

from diffusion_forcing_transformer.torch_utils import bernoulli_tensor
from typing import Optional, Callable, Tuple, Dict
from functools import partial
from tqdm import tqdm
from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np
from utils.configcls import Config

from utils.math.sbto_utils import build_feature_layout, get_feature_indices

# =========================
# Conditioning Embeddings
# =========================
def apply_condition_dropout(
    cond,
    dropout_prob: float,
):
    """
    Apply per-sample dropout for classifier-free guidance.

    Args:
        cond: torch.Tensor or None, shape (B, ...)
        dropout_prob: Probability of dropping condition

    Returns:
        dropped_cond
    """
    if cond is None or dropout_prob <= 0.0:
        return cond

    if not torch.is_tensor(cond):
        cond = torch.as_tensor(cond)

    batch_size = cond.shape[0]

    # Generate mask: 1 keep, 0 drop
    keep_prob = 1.0 - dropout_prob
    
    # Mask shape (B, 1, ...) to broadcast
    mask_shape = (batch_size,) + (1,) * (cond.ndim - 1)
    
    mask = torch.bernoulli(torch.full(mask_shape, keep_prob, device=cond.device))
    
    return cond * mask

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


class DFoTTrajectory(nn.Module):
    def __init__(self, model_config, data_config, noise_scheduler_config=None, training_config=None):
        super().__init__()
        self.model_config = model_config
        self.data_config = data_config
        self.noise_scheduler_config = noise_scheduler_config 
        self.training_config = training_config
        
        self.x_shape = torch.Size([data_config['num_features']])
        self.state_condition = model_config.get("state_condition", False)
        self.task_condition = model_config.get("task_condition", False)
        
        self.state_dim = model_config['hidden_size'] if self.state_condition else 0
        self.task_dim = model_config.get("hidden_size", 64) if self.task_condition else 0

        self.external_cond_dim = self.state_dim + self.task_dim
            
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
                "pos_emb_type": model_config.get("pos_emb_type", "learned_1d"),
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
        beta_schedule = self.noise_scheduler_config.get("beta_schedule", "sigmoid")

        # Map prediction type
        prediction_type = self.noise_scheduler_config.get("prediction_type", "v_prediction")
        objective = "pred_v" if prediction_type == "v_prediction" else "pred_x0" if prediction_type == "sample" else "pred_noise"

        # Diffusion config
        diffusion_cfg = Config({
            "timesteps": self.noise_scheduler_config.get("train_timesteps", 1000),
            "sampling_timesteps": self.noise_scheduler_config.get("inference_timesteps", 50),
            "beta_schedule": beta_schedule,
            "schedule_fn_kwargs": {"shift": 1.0},
            "objective": objective,
            # "loss_weighting": {
            #     "strategy": "goal-weighted",
            #     "final_frame_weight": 5.0,
            # },
            "loss_weighting": {
                "strategy": "fused_min_snr",
                "snr_clip": 5.0,
                "cum_snr_decay": 0.9,
            },
            "ddim_sampling_eta": 0.0,
            "clip_noise": 20.0,
            "use_causal_mask": self.noise_scheduler_config.get("use_causal_mask", False),
        })

        self.max_tokens = data_config['num_timesteps'] // data_config.get('downsample', 1)

        betas=None
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
        
        # In-betweening (masked trajectory completion) config
        self.inbetweening_cfg = training_config.get("inbetweening", {}) if training_config else {}
        self.inbetweening_enabled = self.inbetweening_cfg.get("enabled", False)
        if self.inbetweening_enabled:
            print(f"In-betweening mode enabled: "
                  f"keyframes={self.inbetweening_cfg.get('min_keyframes',2)}-{self.inbetweening_cfg.get('max_keyframes',4)}, "
                  f"keep_first={self.inbetweening_cfg.get('always_keep_first', True)}, "
                  f"keep_last={self.inbetweening_cfg.get('keep_last_partial', False)}")

        # Build feature index map from data config's feature_order
        _layout = build_feature_layout()
        self.feature_index_map = {}
        fidx = 0
        for key in data_config.get("feature_order", []):
            dim = _layout.get(key, 0)
            if dim > 0:
                self.feature_index_map[key] = slice(fidx, fidx + dim)
                fidx += dim

        # Build feature group slices for partial masking
        partial_cfg = self.inbetweening_cfg.get("partial_masking", {})

        # Register joints sub-slices (joints_lower, joints_upper) so they can
        # be referenced as independent feature groups in the config.
        # joints_split: number of leading joint dims treated as "lower body".
        # Default: 12 (6 left-leg + 6 right-leg DOFs come first in the G1 layout).
        joints_split = partial_cfg.get("joints_split", None)
        if joints_split is not None and "joints" in self.feature_index_map:
            j_sl = self.feature_index_map["joints"]
            self.feature_index_map["joints_lower"] = slice(j_sl.start,
                                                            j_sl.start + joints_split)
            self.feature_index_map["joints_upper"] = slice(j_sl.start + joints_split,
                                                            j_sl.stop)

        self.feature_group_slices = {}
        for group_name, keys in partial_cfg.get("feature_groups", {}).items():
            slices = []
            for key in keys:
                if key in self.feature_index_map:
                    slices.append(self.feature_index_map[key])
            if slices:
                self.feature_group_slices[group_name] = slices
        self.group_keep_probs = partial_cfg.get("group_keep_prob", {})
        self.waypoint_noise_fraction = self.inbetweening_cfg.get("waypoint_noise_fraction", 0.25)

        # Waypoint indicator embedding: projects per-feature binary mask (D) into
        # the backbone's hidden space so the model knows which features are constrained.
        # Added to token embeddings after input_embedder + pos_emb, before transformer blocks.
        hidden = model_config.get("hidden_size", 256)

        if self.inbetweening_enabled:
            self.waypoint_indicator_proj = nn.Sequential(
                nn.Linear(data_config['num_features'], hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            # Zero-init the last layer so the indicator is a no-op at the start of training
            nn.init.zeros_(self.waypoint_indicator_proj[-1].weight)
            nn.init.zeros_(self.waypoint_indicator_proj[-1].bias)

            # RePaint resampling config (inference-time back-and-forth loops)
            self.resample_steps = self.inbetweening_cfg.get("resample_steps", 0)


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

    def _generate_inbetweening_mask(self, batch_size: int, n_tokens: int, device: torch.device) -> torch.Tensor:
        """
        Generate a binary mask for in-betweening training.
        mask=1 means KEEP (keyframe), mask=0 means MASK (to be predicted).

        Returns: (B, T) bool tensor
        """
        cfg = self.inbetweening_cfg
        min_kf = cfg.get("min_keyframes", 2)
        max_kf = cfg.get("max_keyframes", 4)
        keep_first = cfg.get("always_keep_first", True)
        keep_last = cfg.get("keep_last_partial", False)

        mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # Determine how many keyframes this sample gets
            n_kf = torch.randint(min_kf, max_kf + 1, (1,)).item()

            forced = []
            if n_kf > 0:
                if keep_first:
                    forced.append(0)
                if n_kf > 1 and keep_last:
                    forced.append(n_tokens - 1)

                # Remaining keyframes chosen randomly from the rest
                available = list(set(range(n_tokens)) - set(forced))
                n_random = max(0, n_kf - len(forced))
                n_random = min(n_random, len(available))

                if n_random > 0:
                    perm = torch.randperm(len(available))
                    chosen = [available[perm[i].item()] for i in range(n_random)]
                else:
                    chosen = []

                all_kf = forced + chosen
                mask[b, all_kf] = True

        return mask
    
    #### ============== ####

    def _generate_waypoint_mask(self, B, T, D, device):
        """
        Generate a feature-level waypoint mask for inbetweening training.

        For each sample, randomly selects keyframe time steps, then for each
        keyframe decides whether it's "full" (all features known) or "partial"
        (only certain feature groups known, chosen randomly per group).

        Returns:
            waypoint_mask: (B, T, D) bool — True = known/clean, False = to be predicted
        """
        cfg = self.inbetweening_cfg
        partial_cfg = cfg.get("partial_masking", {})
        partial_enabled = partial_cfg.get("enabled", False)
        partial_prob = partial_cfg.get("prob", 0.5)

        # Step 1: Frame-level keyframe selection
        frame_mask = self._generate_inbetweening_mask(B, T, device)  # (B, T) bool

        if not partial_enabled:
            # All waypoints are full keyframes (all features known)
            return frame_mask.unsqueeze(-1).expand(-1, -1, D).clone()

        # Step 2: For each keyframe, decide full vs partial
        is_partial = (torch.rand(B, T, device=device) < partial_prob) & frame_mask

        # Force first/last to be full keyframes if configured
        if cfg.get("always_keep_first", True):
            is_partial[:, 0] = False
        if cfg.get("keep_last_partial", False) and T > 0:
            is_partial[:, -1] = True

        is_full = frame_mask & ~is_partial

        # Step 3: Build feature mask
        waypoint_mask = torch.zeros(B, T, D, dtype=torch.bool, device=device)

        # Full keyframes: all features known
        waypoint_mask = waypoint_mask | is_full.unsqueeze(-1)  # broadcast (B,T,1) -> (B,T,D)

        # Partial keyframes: per-group random keep
        if is_partial.any():
            for group_name, slices in self.feature_group_slices.items():
                keep_prob = self.group_keep_probs.get(group_name, 0.5)
                keep_group = (torch.rand(B, T, device=device) < keep_prob) & is_partial
                for s in slices:
                    width = s.stop - s.start
                    waypoint_mask[:, :, s] |= keep_group.unsqueeze(-1).expand(-1, -1, width)

        return waypoint_mask

    def _inject_waypoints(
        self,
        xs: torch.Tensor,
        waypoint_values: torch.Tensor,
        waypoint_mask: torch.Tensor,
        noise_levels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inject waypoint values into xs with noise-aware handling.

        - Fully-known frames (all features marked): always inject clean values.
        - Partially-known frames: if noise_levels is provided, inject q_sampled
          values (RePaint-style) so partial waypoints stay consistent with the
          current denoising noise level; otherwise inject clean values.

        Args:
            xs: (B, T, D) current noised/denoised predictions
            waypoint_values: (B, T, D) clean target values
            waypoint_mask: (B, T, D) bool — True = known feature
            noise_levels: (B, T) int noise levels for RePaint injection, or None

        Returns:
            xs with waypoint features injected
        """
        if waypoint_values is None or waypoint_mask is None:
            return xs

        wm = waypoint_mask.to(xs.device)
        wv = waypoint_values.to(xs.device, dtype=xs.dtype)

        fully_known = wm.all(dim=-1)  # (B, T)

        # Full keyframes: always inject clean values
        xs = torch.where(fully_known.unsqueeze(-1) & wm, wv, xs)

        # Partial waypoints
        partially_known = wm.any(dim=-1) & ~fully_known  # (B, T)
        if noise_levels is not None and partially_known.any():
            # RePaint-style: noise known features to match current noise level
            k_wp = torch.clamp(noise_levels, min=0)  # (B, T)
            wp_noise = torch.randn_like(wv)
            wp_noise = torch.clamp(wp_noise, -self.clip_noise, self.clip_noise)
            noised_wv = self.diffusion_model.q_sample(x_start=wv, k=k_wp, noise=wp_noise)
            # At fully-denoised steps (noise_level == -1), inject clean values
            is_clean = (noise_levels == -1).unsqueeze(-1)  # (B, T, 1)
            injection = torch.where(is_clean, wv, noised_wv)
            xs = torch.where(
                partially_known.unsqueeze(-1) & wm, injection, xs
            )
        else:
            # No noise levels — inject clean for all remaining waypoints
            xs = torch.where(wm & ~fully_known.unsqueeze(-1), wv, xs)

        return xs

    def _model_predictions_with_indicator(
        self, x, k, external_cond=None, external_cond_mask=None,
        waypoint_mask=None,
    ):
        """
        Run model predictions while injecting the waypoint indicator embedding
        into the backbone's hidden representation.
        """
        from diffusion_forcing_transformer.discrete_diffusion import ModelPrediction

        backbone = self.diffusion_model.model  # DiT1D

        # 1. Input embedding + positional embedding (standard DiT1D path)
        h = backbone.input_embedder(x)          # (B, T, hidden)
        h = backbone.pos_emb(h)

        # 2. Inject waypoint indicator
        if waypoint_mask is not None:
            # waypoint_mask: (B, T, D) bool -> float -> project to hidden
            indicator = self.waypoint_indicator_proj(waypoint_mask.float().to(x.device))  # (B, T, hidden)
            h = h + indicator

        # 3. Condition embedding (same as DiT1D.forward)
        c = backbone.noise_level_pos_embedding(k)
        if external_cond is not None:
            c = c + backbone.external_cond_embedding(external_cond, external_cond_mask)

        # 4. Transformer blocks
        for block in backbone.blocks:
            h = backbone._checkpoint(block, h, c)

        # 5. Final layer
        model_output = backbone.final_layer(h, c)

        # 6. Convert to predictions (same logic as DiscreteDiffusion.model_predictions)
        dm = self.diffusion_model
        if dm.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -dm.clip_noise, dm.clip_noise)
            x_start = dm.predict_start_from_noise(x, k, pred_noise)
        elif dm.objective == "pred_x0":
            x_start = model_output
            pred_noise = dm.predict_noise_from_start(x, k, x_start)
        elif dm.objective == "pred_v":
            v = model_output
            x_start = dm.predict_start_from_v(x, k, v)
            pred_noise = dm.predict_noise_from_v(x, k, v)
        else:
            raise ValueError(f"Unknown objective: {dm.objective}")

        return ModelPrediction(pred_noise, x_start, model_output)

    def _sample_step_with_indicator(
        self, x, curr_noise_level, next_noise_level, external_cond,
        external_cond_mask=None, guidance_fn=None, guidance_goal=None,
        guidance_wt=1.0, cfg_w=1.0, waypoint_mask=None,
    ):
        """
        DDIM sample step that uses the waypoint indicator embedding.
        Mirrors DiscreteDiffusion.ddim_sample_step but with the indicator injected.
        """
        dm = self.diffusion_model
        clipped_curr = torch.clamp(curr_noise_level, min=0)

        alpha = dm.alphas_cumprod[clipped_curr]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            dm.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            dm.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c_coeff = (1 - alpha_next - sigma**2).sqrt()

        alpha = dm.add_shape_channels(alpha)
        alpha_next = dm.add_shape_channels(alpha_next)
        c_coeff = dm.add_shape_channels(c_coeff)
        sigma = dm.add_shape_channels(sigma)

        # Conditional prediction (with indicator)
        pred_cond = self._model_predictions_with_indicator(
            x, clipped_curr, external_cond, external_cond_mask=None,
            waypoint_mask=waypoint_mask,
        )

        # Unconditional prediction (with indicator but masked external cond)
        masked_ext = external_cond_mask * external_cond if external_cond is not None else None
        pred_uncond = self._model_predictions_with_indicator(
            x, clipped_curr, masked_ext, external_cond_mask=None,
            waypoint_mask=waypoint_mask,
        )

        # CFG
        x_start = pred_uncond.pred_x_start + cfg_w * (
            pred_cond.pred_x_start - pred_uncond.pred_x_start
        )
        pred_noise = pred_uncond.pred_noise + cfg_w * (
            pred_cond.pred_noise - pred_uncond.pred_noise
        )

        if cfg_w > 1.0:
            s = torch.quantile(torch.abs(x_start).flatten(1), 0.995, dim=1)
            s = torch.maximum(s, torch.ones_like(s)).view(-1, 1, 1)
            x_start = torch.clamp(x_start, -s, s) / s

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -dm.clip_noise, dm.clip_noise)
        x_pred = x_start * alpha_next.sqrt() + pred_noise * c_coeff + sigma * noise

        # Only update frames where noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(dm.add_shape_channels(mask), x, x_pred)
        return x_pred

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
                s_cond = s_cond.unsqueeze(1)
            s_cond = apply_condition_dropout(
                s_cond, 
                self.training_config.get("condition_dropout_prob", {}).get("state", 0.0),
            )
            cond_list.append(s_cond)
        if self.task_condition and task_cond is not None:
            t_cond = self.task_embedding(task_cond)
            if t_cond.ndim == 2:
                t_cond = t_cond.unsqueeze(1)
            t_cond = apply_condition_dropout(
                t_cond, 
                self.training_config.get("condition_dropout_prob", {}).get("task", 0.0),
            )
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

        # --- Generate noise levels ---
        B, T, D = x.shape
        if timesteps is None:
            k, masks = self._get_training_noise_levels(x, masks)
        else:
            k = timesteps

        # --- Waypoint masking (inbetweening) ---
        waypoint_mask = None  # (B, T, D) bool or None
        if self.training and self.inbetweening_enabled:
            waypoint_mask = self._generate_waypoint_mask(B, T, D, x.device)
            fully_known = waypoint_mask.all(dim=-1)  # (B, T)
            partially_known = waypoint_mask.any(dim=-1) & ~fully_known  # (B, T)

            # Fully-known frames → noise_level = 0 (clean signal to model)
            k = torch.where(fully_known, torch.zeros_like(k), k)

            # Partially-known frames → intermediate noise level
            if partially_known.any():
                max_wp_k = max(int(self.waypoint_noise_fraction * self.timesteps), 1)
                wp_k = torch.randint(0, max_wp_k, k.shape, device=k.device)
                k = torch.where(partially_known, wp_k, k)

        # --- Forward diffusion: q_sample ---
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        noised_x = self.diffusion_model.q_sample(x_start=x, k=k, noise=noise)

        # Inject known features into noised input
        if waypoint_mask is not None:
            noised_x = self._inject_waypoints(noised_x, x, waypoint_mask)

        # --- Waypoint indicator: inject into backbone hidden space ---
        # We hook into the DiT1D forward: apply input_embedder + pos_emb, add
        # indicator embedding, then run transformer blocks + final_layer.
        # This replaces the standard model_predictions call when waypoints are active.
        if waypoint_mask is not None and waypoint_mask.any():
            model_pred = self._model_predictions_with_indicator(
                noised_x, k, ext_cond, external_cond_mask=None,
                waypoint_mask=waypoint_mask,
            )
        else:
            model_pred = self.diffusion_model.model_predictions(
                x=noised_x, k=k, external_cond=ext_cond, external_cond_mask=None
            )
        x_pred = model_pred.pred_x_start

        # --- Loss ---
        if self.diffusion_model.objective == "pred_noise":
            target = noise
        elif self.diffusion_model.objective == "pred_x0":
            target = x
        elif self.diffusion_model.objective == "pred_v":
            target = self.diffusion_model.predict_v(x, k, noise)
        else:
            raise ValueError(f"Unknown objective: {self.diffusion_model.objective}")

        loss = F.mse_loss(model_pred.model_out, target.detach(), reduction="none")

        # Diffusion loss weighting (min-SNR, etc.)
        loss_weight = self.diffusion_model.compute_loss_weights(
            k, self.diffusion_model.loss_weighting.strategy
        )
        loss_weight = self.diffusion_model.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        # Zero out loss on known/pinned features so the model isn't
        # penalised for predicting clean data it was given (esp. k=0 frames
        # where the v-prediction target is pure noise → random loss).
        fully_known = waypoint_mask.all(dim=-1)  # (B, T)
        if fully_known.any():
            loss = loss * (~fully_known).float().unsqueeze(-1)  

        # Apply frame-level validity masks
        loss = self._reweight_loss(loss, masks)

        return {"loss": loss, "xs_pred": x_pred, "xs": x}

    def sample(self, num_trajectories, model_cond=None, cfg_w=0.0, guidance_wt=1.0, guidance_goal=None, inpaint=False, no_state_cond=False, waypoint_values=None, waypoint_mask=None):
        """
        Sampling method. Supports optional feature-level waypoint injection for
        guided generation. When waypoint_values/waypoint_mask are provided,
        known features are injected at each denoising step.
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
            self.state_cond = state_cond_input
            s_cond = self.state_embedding(state_cond_input) # (B, C)
            if no_state_cond:
                s_cond = apply_condition_dropout(
                    s_cond, 
                    1.0
                )
            cond_list.append(s_cond)
        if self.task_condition and task_cond is not None:
            self.task_cond = task_cond
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

        # ---- Iterative diffusion sampling with optional waypoint injection ----
        xs_pred, _ = self._sample_sequence(
            batch_size=num_trajectories,
            conditions=conditions,
            task_cond=task_cond,
            guidance_wt=guidance_wt,
            guidance_goal=guidance_goal,
            guidance_fn=guidance_goal_mse if (guidance_wt > 0 and guidance_goal is not None) else None,
            inpaint=inpaint,
            cfg_w=cfg_w,
            waypoint_values=waypoint_values,
            waypoint_mask=waypoint_mask,
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
        task_cond: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        guidance_goal: Optional[torch.Tensor] = None,
        guidance_wt: float = 0.0,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
        inpaint: bool = False,
        cfg_w: float = 1.0,
        waypoint_values: Optional[torch.Tensor] = None,
        waypoint_mask: Optional[torch.Tensor] = None,
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
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        # Inject waypoint values into initial noise (feature-level mask)
        if waypoint_mask is not None and waypoint_mask.any():
            # Use the first row of the scheduling matrix (max noise) so partial
            # waypoints are noised to match the starting noise level (not clean).
            init_sched = self._generate_scheduling_matrix(
                horizon - padding, padding,
            ).to(xs_pred.device)
            init_noise_levels = repeat(init_sched[0], "t -> b t", b=batch_size)
            xs_pred = self._inject_waypoints(
                xs_pred, waypoint_values, waypoint_mask,
                noise_levels=init_noise_levels,
            )

        # create empty context and zero context mask
        context_mask = torch.zeros(
            (batch_size, horizon), dtype=torch.long, device=xs_pred.device
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

            external_conditions_mask = torch.zeros(
                (batch_size, 1, self.external_cond_dim), dtype=torch.bool, device=xs_pred.device
            )

            external_conditions_mask[..., :self.task_dim] = 1.0

            # Choose indicator-aware or standard sample step
            # Must match training condition: use indicator whenever waypoint_mask is present
            use_indicator = waypoint_mask is not None and waypoint_mask.any()

            def _do_sample_step(x_in, from_k, to_k):
                if use_indicator:
                    return self._sample_step_with_indicator(
                        x_in, from_k, to_k, conditions,
                        external_cond_mask=external_conditions_mask,
                        guidance_fn=guidance_fn, guidance_goal=guidance_goal,
                        guidance_wt=guidance_wt, cfg_w=cfg_w,
                        waypoint_mask=waypoint_mask,
                    )
                else:
                    return self.diffusion_model.sample_step(
                        x_in, from_k, to_k, conditions,
                        external_conditions_mask,
                        guidance_fn=guidance_fn, guidance_goal=guidance_goal,
                        guidance_wt=guidance_wt, cfg_w=cfg_w,
                    )

            xs_pred = _do_sample_step(xs_pred, from_noise_levels, to_noise_levels)

            # Re-inject waypoint values (RePaint-style for partial waypoints)
            if waypoint_mask is not None and waypoint_mask.any():
                xs_pred = self._inject_waypoints(
                    xs_pred, waypoint_values, waypoint_mask,
                    noise_levels=to_noise_levels,
                )

                # --- RePaint resampling: time-travel back-and-forth ---
                # Re-noise to from_noise_levels, re-denoise to to_noise_levels,
                # re-inject waypoints. Repeat to harmonize constrained features
                # with unconstrained ones.
                resample_n = self.resample_steps if use_indicator else 0
                # Only resample while we are still at meaningful noise levels
                still_noisy = (from_noise_levels.max() > 0).item()
                if resample_n > 0 and still_noisy:
                    for i in range(resample_n):
                        # 1. Re-noise back to from_noise_levels
                        re_noise = torch.randn_like(xs_pred)
                        re_noise = torch.clamp(re_noise, -self.clip_noise, self.clip_noise)
                        clamped_from = torch.clamp(from_noise_levels, min=0)
                        xs_pred = self.diffusion_model.q_sample(
                            x_start=xs_pred, k=clamped_from, noise=re_noise
                        )
                        # 2. Re-inject waypoints at from_noise_level
                        xs_pred = self._inject_waypoints(
                            xs_pred, waypoint_values, waypoint_mask,
                            noise_levels=from_noise_levels,
                        )
                        # 3. Re-denoise
                        xs_pred = _do_sample_step(xs_pred, from_noise_levels, to_noise_levels)
                        # 4. Re-inject waypoints at to_noise_level
                        if i != resample_n - 1:  # no need to re-inject on last iteration
                            xs_pred = self._inject_waypoints(
                                xs_pred, waypoint_values, waypoint_mask,
                                noise_levels=to_noise_levels,
                            )
                
            if inpaint:
                f_map = get_feature_indices(build_feature_layout())
                xs_pred[..., 0, f_map['joints']] = self.state_cond[..., 0, :29]
                xs_pred[..., 0, f_map['body_z']] = self.state_cond[..., 0, 29:30]
                xs_pred[..., 0, f_map['body_rot6d']] = self.state_cond[..., 0, 30:36]
                xs_pred[..., 0, f_map['obj_rel_pos']] = self.state_cond[..., 0, 36:39]
                xs_pred[..., 0, f_map['obj_rel_rot6d']] = self.state_cond[..., 0, 39:45]
            
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record
