import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

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
            
        self.goal_condition = self.model_cfg.get("goal_condition", False)
        self.state_condition = self.model_cfg.get("state_condition", False)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_schedule_cfg["train_timesteps"], 
            beta_schedule=self.noise_schedule_cfg["beta_schedule"],
            prediction_type=self.noise_schedule_cfg["prediction_type"],
        )

        self.deterministic_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_schedule_cfg["train_timesteps"], 
            beta_schedule=self.noise_schedule_cfg["beta_schedule"],
            prediction_type=self.noise_schedule_cfg["prediction_type"], 
        )

        if model_type == "simple_diffusion":
            self.model = SimpleDiffuser(inp_size=self.input_size, num_channels=self.num_channels).to(self.device)
        elif model_type == "latent_diffusion":
            self.model = UNetDiffuser(inp_size=self.input_size, num_channels=self.num_channels).to(self.device)
        elif model_type == "dfot":
            self.model = DFoTTrajectory(self.model_cfg, self.data_cfg, self.noise_schedule_cfg, self.training_cfg, self.noise_scheduler).to(self.device)
        elif model_type == "dit1d_wrapper":
            self.model = SimpleTrajectoryDiffuser(self.model_cfg, self.data_cfg, self.noise_schedule_cfg, self.noise_scheduler).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")           
        
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        
    def loadWeights(self, policy_num, ema=False):
        if ema:
            path=self.save_dir+f"ema_model_{policy_num}.pth"
        else:
            path=self.save_dir+f"model_{policy_num}.pth"
        print(f"Loading model weights from {path}... \n")
        weights = torch.load(path, map_location=self.device)
        if "model" in weights:
            self.model.load_state_dict(weights["model"])
            return weights
        else:
            self.model.load_state_dict(weights)
            return None

    def load_weights_from_file(self, path):
        """
        Load weights from a specific file path.
        """
        print(f"Loading model weights from {path}... \n")
        weights = torch.load(path, map_location=self.device)
        if "model" in weights:
            self.model.load_state_dict(weights["model"])
        else:
            self.model.load_state_dict(weights)

    def getSample(self, num_trajectories=1, state_cond=None, goal_cond=None, deterministic=False, cfg_w=1.0):
        """
        Run reverse diffusion to generate trajectories.
        """
        if deterministic:
            print("Using DDIM Scheduler for inference...")
            inference_noise_scheduler = self.deterministic_noise_scheduler
        else:
            print("Using DDPM Scheduler for inference...")
            inference_noise_scheduler = self.noise_scheduler

        inference_noise_scheduler.set_timesteps(self.noise_schedule_cfg["inference_timesteps"])
        sample = torch.randn(num_trajectories, self.num_channels, self.input_size).to(self.device)
        
        cond_list = []
        if state_cond is not None:
            if isinstance(state_cond, np.ndarray):
                state_cond = torch.from_numpy(state_cond).to(self.device).to(sample.dtype)
            cond_list.append(state_cond)

        if goal_cond is not None:
            if isinstance(goal_cond, np.ndarray):
                goal_cond = torch.from_numpy(goal_cond).to(self.device).to(sample.dtype)
            cond_list.append(goal_cond)

        # tuple containing state cond, goal cond    
        model_cond = tuple(cond_list) if len(cond_list) > 0 else None
        if len(cond_list) == 1: model_cond = cond_list[0]

        sample_kwargs = {}
        if self.model_cfg['type'] == "dit1d_wrapper":
            sample_kwargs['scheduler'] = inference_noise_scheduler

        sample = self.model.sample(
            num_trajectories, 
            model_cond=model_cond,
            cfg_w=cfg_w,
            **sample_kwargs
        )
        
        return sample.detach()

    
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