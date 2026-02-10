import yaml
import os 

def load_config(config_path: str, auto_conf: bool = False) -> dict:
    """Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        tuple: (model_cfg, data_cfg, training_cfg)
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    noise_sched_cfg = config.get('noise_scheduler', {})
    
    save_dir = get_save_path(model_cfg, data_cfg, training_cfg)
    saved_config_path = os.path.join(save_dir, "config.yaml")
    
    # Check for saved config in the run directory
    if auto_conf and os.path.exists(saved_config_path):
        print(f"Loading configuration from {saved_config_path}...\n")
        model_cfg, data_cfg, training_cfg, noise_sched_cfg = load_config(saved_config_path, False)

    return model_cfg, data_cfg, training_cfg, noise_sched_cfg

def get_run_path(model_cfg: dict, data_cfg: dict, training_cfg: dict) -> str:
    input_size = data_cfg.get('num_timesteps', 'unknown')
    num_channels = data_cfg.get('num_features', 'unknown')
    suffix = training_cfg.get('suffix', '')
    save_dir = f"{model_cfg.get('type', 'model')}_ts{input_size}_f{num_channels}"
    save_dir += f"{suffix}/"
    return save_dir

def get_save_path(model_cfg: dict, data_cfg: dict, training_cfg: dict) -> str:
    """Construct the save directory path based on configurations.
    Args:
        model_cfg (dict): Model configuration dictionary.
        data_cfg (dict): Data configuration dictionary.
        training_cfg (dict): Training configuration dictionary.
    
    Returns:
        str: Constructed save directory path.
    """
    run_path = get_run_path(model_cfg=model_cfg, data_cfg=data_cfg, training_cfg=training_cfg)
    save_dir = training_cfg.get('save_dir', './runs') + run_path
    return save_dir

def get_log_path(model_cfg: dict, data_cfg: dict, training_cfg: dict) -> str:
    """Construct the log directory path based on configurations.
    Args:
        model_cfg (dict): Model configuration dictionary.
        data_cfg (dict): Data configuration dictionary.
        training_cfg (dict): Training configuration dictionary.
    
    Returns:
        str: Constructed log directory path.
    """
    run_path = get_run_path(model_cfg=model_cfg, data_cfg=data_cfg, training_cfg=training_cfg)
    log_dir = training_cfg.get('log_dir', './logs/') + run_path
    print(f"Saving logs to {log_dir}...\n")
    return log_dir

def get_data_path(data_cfg: dict) -> str:
    data_path = data_cfg.get("dir_path", "/home/") + data_cfg.get("train_path", "")
    path_suffix = ''
    if data_cfg.get("rot_type", "quat") != "quat":
        path_suffix = f"_{data_cfg['rot_type']}"
        
    data_path = data_path.replace(".npz", f"{path_suffix}.npz")
    print(f"Getting data from {data_path}...\n")
    return data_path

def get_norm_path(model_cfg: dict, training_cfg: dict, data_cfg: dict) -> str:    
    run_path = get_run_path(model_cfg=model_cfg, data_cfg=data_cfg, training_cfg=training_cfg)
    save_dir = training_cfg.get('save_dir', './runs') + run_path
    norm_path = os.path.join(save_dir, "norm_stats.npz")
    
    return norm_path
