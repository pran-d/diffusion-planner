import torch
import torch.nn as nn
import numpy as np
from typing import Iterable
from contextlib import nullcontext

from .latent_diffusion import UNetDiffuser
from .basic_diffusion import SimpleDiffuser
from .simple_trajectory import SimpleTrajectoryDiffuser
from .dfot_trajectory import DFoTTrajectory

from config.configure import get_save_path
from diffusers import DDPMScheduler, DDIMScheduler


class RobotDiffuser():
    """
    Interfacing class for the different diffusion models in this project.
    """

    def __init__(self, model_config, data_config, training_config, noise_scheduler_config, mode, device):
        self.model_cfg = model_config
        self.data_cfg = data_config
        self.training_cfg = training_config
        self.noise_schedule_cfg = noise_scheduler_config
        self.device = device

        self.save_dir = get_save_path(model_config, data_config, training_config)

        model_type = self.model_cfg['type']
        self.input_size = data_config['num_timesteps'] // data_config.get('downsample', 1)
        
        self.num_channels = data_config['num_features']
            
        self.state_condition = self.model_cfg.get("state_condition", False)

        if model_type == "simple_diffusion":
            self.model = SimpleDiffuser(inp_size=self.input_size, num_channels=self.num_channels).to(self.device)
        elif model_type == "latent_diffusion":
            self.model = UNetDiffuser(inp_size=self.input_size, num_channels=self.num_channels).to(self.device)
        elif model_type == "dfot":
            self.model = DFoTTrajectory(self.model_cfg, self.data_cfg, self.noise_schedule_cfg, self.training_cfg).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")           

        self.use_fp16 = bool(self.model_cfg.get("use_fp16", False))
        self.compile_model = bool(self.model_cfg.get("compile_model", False))
        self._compiled_once = False

        if self.compile_model and hasattr(torch, "compile"):
            try:
                backbone = None
                if hasattr(self.model, "diffusion_model") and hasattr(self.model.diffusion_model, "model"):
                    backbone = self.model.diffusion_model.model
                if backbone is not None:
                    self.model.diffusion_model.model = torch.compile(backbone, mode="reduce-overhead")
                    self._compiled_once = True
                    print("Enabled torch.compile on diffusion backbone (mode=reduce-overhead).")
            except Exception as e:
                print(f"[Warning] torch.compile failed, continuing without compile: {e}")
        
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        
    def loadWeights(self, policy_num, ema=False):
        if isinstance(policy_num, str):
            path = policy_num
        else:
            if ema:
                path = self.save_dir + f"ema_model_{policy_num}.pth"
            else:
                path = self.save_dir + f"model_{policy_num}.pth"
        print(f"Loading diffusion model weights from {path}{'  [EMA]' if ema else ''}... \n")
        weights = torch.load(path, map_location=self.device, weights_only=False)
        # EMA weight files are saved as plain state_dicts (no "model" key)
        state = weights["model"] if "model" in weights else weights
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [Warning] Missing keys (will use init values): {missing}")
        if unexpected:
            print(f"  [Warning] Unexpected keys (ignored): {unexpected}")
        # Return the full checkpoint dict only for non-EMA files that have it
        if ema:
            return None
        return weights if "model" in weights else None

    def load_weights_from_file(self, path):
        """Load weights from a specific file path."""
        print(f"Loading model weights from {path}... \n")
        weights = torch.load(path, map_location=self.device, weights_only=False)
        state = weights["model"] if "model" in weights else weights
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [Warning] Missing keys (will use init values): {missing}")
        if unexpected:
            print(f"  [Warning] Unexpected keys (ignored): {unexpected}")

    def getSample(self, num_trajectories=1, state_cond=None, goal_cond=None, deterministic=False, cfg_w=1.0, guidance_wt=0.0, guidance_goal=None, no_state_cond=False, waypoint_values=None, waypoint_mask=None):
        """
        Run reverse diffusion to generate trajectories.
        """
        sample = torch.randn(num_trajectories, self.num_channels, self.input_size).to(self.device)
        
        cond_list = []
        if getattr(self.model, 'state_condition', False) and state_cond is not None:
            if isinstance(state_cond, np.ndarray):
                state_cond = torch.from_numpy(state_cond).to(self.device).to(sample.dtype)
            cond_list.append(state_cond)

        if getattr(self.model, 'task_condition', False) and goal_cond is not None:
            if isinstance(goal_cond, np.ndarray):
                goal_cond = torch.from_numpy(goal_cond).to(self.device).to(sample.dtype)
            cond_list.append(goal_cond)

        if getattr(self.model, 'style_condition', False) and style_cond is not None:
            if isinstance(style_cond, np.ndarray):
                style_cond = torch.from_numpy(style_cond).to(self.device).to(sample.dtype)
            cond_list.append(style_cond)


        # tuple containing state cond, goal cond    
        model_cond = tuple(cond_list) if len(cond_list) > 0 else None
        if len(cond_list) == 1: model_cond = cond_list[0]

        sample_kwargs = {
            'guidance_goal': guidance_goal if guidance_goal is not None else goal_cond, 
            'guidance_wt': guidance_wt, 'inpaint': self.model_cfg.get("inpaint", False),
            'no_state_cond': no_state_cond,
            'waypoint_values': waypoint_values,
            'waypoint_mask': waypoint_mask,
        }

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if (self.use_fp16 and self.device.type == "cuda") else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                sample = self.model.sample(
                    num_trajectories,
                    model_cond=model_cond,
                    cfg_w=cfg_w,
                    **sample_kwargs
                )
        
        return sample.detach()

    def set_inference_speedups(self, use_fp16: bool | None = None, compile_model: bool | None = None):
        if use_fp16 is not None:
            self.use_fp16 = bool(use_fp16)
        if compile_model is not None:
            want_compile = bool(compile_model)
            if want_compile and (not self._compiled_once) and hasattr(torch, "compile"):
                try:
                    backbone = None
                    if hasattr(self.model, "diffusion_model") and hasattr(self.model.diffusion_model, "model"):
                        backbone = self.model.diffusion_model.model
                    if backbone is not None:
                        self.model.diffusion_model.model = torch.compile(backbone, mode="reduce-overhead")
                        self._compiled_once = True
                        print("Enabled torch.compile on diffusion backbone (runtime).")
                except Exception as e:
                    print(f"[Warning] torch.compile failed at runtime: {e}")
            self.compile_model = want_compile

    
    def save_ema_weights(self, parameters: Iterable[torch.nn.Parameter], path):
        curr_model_params = list(self.model.parameters())
        parameters = list(parameters)
        
        # ---- 1. Store original model params ----
        original_params = [p.detach().cpu().clone() for p in curr_model_params]

        # ---- 2. Copy EMA params into the model ----
        with torch.no_grad():
            for model_p, ema_p in zip(curr_model_params, parameters):
                model_p.copy_(ema_p.detach())

        # ---- 3. Save model in this temporary EMA state ----
        torch.save(self.model.state_dict(), path)

        # ---- 4. Restore the original parameters ----
        with torch.no_grad():
            for model_p, orig_p in zip(curr_model_params, original_params):
                model_p.copy_(orig_p)