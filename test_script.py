import torch
import numpy as np
from config.configure import load_config, get_data_path, get_norm_path
from datasets.flexible_dataset import FlexibleWindowDataset
from utils.data.load_dataset import preload_dataset

model_cfg, data_cfg, training_cfg, scheduler_cfg = load_config("config/config.yaml")

data_path = get_data_path(data_cfg)
norm_path = get_norm_path(model_cfg, training_cfg, data_cfg)

data_buff = preload_dataset(data_cfg, data_path)
dataset = FlexibleWindowDataset(
    data_buffer=data_buff, config=data_cfg, norm_path=norm_path,
    calculate_stats=False, training_cfg=training_cfg,
)

sample = dataset[0]
print(f"future: {sample[0].shape}, current_state: {sample[1].shape}")
print(f"num_features={dataset.num_features}, num_observations={dataset.num_observations}")
