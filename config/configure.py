import yaml
import os

# Resolved path to the config/ directory (so helpers work regardless of cwd)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_PATHS_FILE = os.path.join(_CONFIG_DIR, "paths.yaml")


def load_paths() -> dict:
    """Load machine-local paths from config/paths.yaml.

    Returns the ``paths`` sub-dict so callers can do::

        paths = load_paths()
        save_dir = paths["save_dir"]

    Falls back to an empty dict if the file is missing so that explicit
    config values still work without a paths.yaml.
    """
    if not os.path.exists(_PATHS_FILE):
        return {}
    with open(_PATHS_FILE, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("paths", {})


def recursive_merge(d1, d2):
    """
    Recursively merge dictionary d2 into d1.
    """
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            recursive_merge(d1[k], v)
        else:
            d1[k] = v
    return d1

def load_yaml_with_includes(path):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f) or {}

    if 'includes' in conf:
        includes = conf.pop('includes')
        base_dir = os.path.dirname(path)
        # We start with an empty dict and merge includes in order
        merged_conf = {}
        for inc in includes:
            if not os.path.isabs(inc):
                inc_path = os.path.abspath(inc)
            else:
                inc_path = inc
                
            inc_conf = load_yaml_with_includes(inc_path)
            recursive_merge(merged_conf, inc_conf)
        
        # Finally merge the main config on top
        recursive_merge(merged_conf, conf)
        return merged_conf
    
    return conf

def load_config(config_path: str, auto_conf: bool = False) -> dict:
    """Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        tuple: (model_cfg, data_cfg, training_cfg, noise_sched_cfg)
    """

    config = load_yaml_with_includes(config_path)

    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    noise_sched_cfg = config.get('noise_scheduler', {})

    # Apply paths.yaml as defaults (config values take precedence)
    paths = load_paths()
    if paths:
        data_cfg.setdefault('dir_path', paths.get('data_dir', './'))
        data_cfg.setdefault('train_path', paths.get('train_path', ''))
        if paths.get('task_list_path') is not None:
            data_cfg.setdefault('task_list_path', paths['task_list_path'])
        training_cfg.setdefault('save_dir', paths.get('save_dir', './runs/'))
        training_cfg.setdefault('log_dir', paths.get('log_dir', './logs/'))
        if 'mj_model_xml' in paths:
            config.setdefault('mj_model_xml', paths['mj_model_xml'])
        if 'mj_model_repeated_xml' in paths:
            config.setdefault('mj_model_repeated_xml', paths['mj_model_repeated_xml'])

    # Propagate root-level keys to training_cfg for backward compatibility / convenience
    for key in ['save_dir', 'log_dir', 'suffix']:
        if key in config:
            training_cfg[key] = config[key]

    save_dir = get_save_path(model_cfg, data_cfg, training_cfg)
    saved_config_path = os.path.join(save_dir, "config.yaml")

    # Check for saved config in the run directory
    if auto_conf and os.path.exists(saved_config_path):
        print(f"Loading configuration from {saved_config_path}...\n")
        model_cfg, data_cfg, training_cfg, noise_sched_cfg = load_config(saved_config_path, False)

    return model_cfg, data_cfg, training_cfg, noise_sched_cfg

def get_run_path(model_cfg: dict, data_cfg: dict, training_cfg: dict) -> str:
    input_size = data_cfg.get('num_timesteps', 'unknown') // data_cfg.get('downsample', 1)
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

def get_norm_path(model_cfg: dict, training_cfg: dict, data_cfg: dict, num_training_calls=None) -> str:
    run_path = get_run_path(model_cfg=model_cfg, data_cfg=data_cfg, training_cfg=training_cfg)
    save_dir = training_cfg.get('save_dir', './runs') + run_path
    norm_file_name = f"norm_stats.npz" if num_training_calls is None else f"norm_stats_{num_training_calls}.npz"
    norm_path = os.path.join(save_dir, norm_file_name)
    return norm_path


def get_mj_xml_paths() -> tuple[str, str]:
    """Return the (mj_model_xml, mj_model_repeated_xml) paths from paths.yaml.

    Falls back to the filenames that previously existed in the repo root so
    existing code is unaffected if paths.yaml is not present.
    """
    paths = load_paths()
    xml = paths.get('mj_model_xml', 'mj_model.xml')
    repeated_xml = paths.get('mj_model_repeated_xml', 'mj_model_repeated.xml')
    return xml, repeated_xml
