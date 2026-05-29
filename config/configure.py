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
                inc_path = os.path.abspath(os.path.join(base_dir, inc))
            else:
                inc_path = inc
                
            inc_conf = load_yaml_with_includes(inc_path)
            recursive_merge(merged_conf, inc_conf)
        
        # Finally merge the main config on top
        recursive_merge(merged_conf, conf)
        return merged_conf
    
    return conf

def _load_paths(config_path: str) -> dict:
    """Load machine-local paths from paths.yaml next to the given config file.

    The file is optional — if it doesn't exist an empty dict is returned.
    Returns the nested ``paths:`` mapping (or top-level keys if there is no
    ``paths:`` wrapper).
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    paths_file = os.path.join(config_dir, "paths.yaml")
    if not os.path.exists(paths_file):
        paths_file = _PATHS_FILE
        
    if not os.path.exists(paths_file):
        return {}
    with open(paths_file, "r") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("paths", raw)


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

    # ── Inject machine-local paths from paths.yaml as defaults ──────────
    paths = _load_paths(config_path)
    # Data paths
    if data_cfg.get("dir_path") is None:
        data_cfg["dir_path"] = paths.get("data_dir", "./")
    if data_cfg.get("train_path") is None:
        data_cfg["train_path"] = paths.get("train_path", "")
    if data_cfg.get("task_list_path") is None:
        data_cfg["task_list_path"] = paths.get("task_list_path", "")
    if data_cfg.get("file_names") is None:
        data_cfg["file_names"] = paths.get("file_names", None)
                
    # Training / run paths
    if training_cfg.get("save_dir") is None:
        training_cfg["save_dir"] = paths.get("save_dir", "./runs/")
    if training_cfg.get("log_dir") is None:
        training_cfg["log_dir"] = paths.get("log_dir", "./logs/")

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
    # Some ablation configs are partial overrides (e.g. training-only), so
    # model/data keys may be missing. Fill missing run-name fields from the
    # base config/config.yaml when available.
    model_type = model_cfg.get('type', None)
    num_timesteps = data_cfg.get('num_timesteps', None)
    downsample = data_cfg.get('downsample', None)
    num_channels = data_cfg.get('num_features', None)

    if model_type is None or num_timesteps is None or downsample is None or num_channels is None:
        base_cfg_path = os.path.join(_CONFIG_DIR, "config.yaml")
        if os.path.exists(base_cfg_path):
            try:
                base = load_yaml_with_includes(base_cfg_path) or {}
                base_model = base.get("model", {}) if isinstance(base.get("model", {}), dict) else {}
                base_data = base.get("data", {}) if isinstance(base.get("data", {}), dict) else {}
                if model_type is None:
                    model_type = base_model.get("type", None)
                if num_timesteps is None:
                    num_timesteps = base_data.get("num_timesteps", None)
                if downsample is None:
                    downsample = base_data.get("downsample", None)
                if num_channels is None:
                    num_channels = base_data.get("num_features", None)
            except Exception:
                pass

    try:
        num_timesteps_i = int(num_timesteps)
        downsample_i = max(1, int(downsample))
        input_size = num_timesteps_i // downsample_i
    except (TypeError, ValueError):
        input_size = 'unknown'

    try:
        num_channels = int(num_channels)
    except (TypeError, ValueError):
        num_channels = 'unknown'

    model_type = model_type if isinstance(model_type, str) and model_type else 'model'
    suffix = training_cfg.get('suffix', '')
    save_dir = f"{model_type}_ts{input_size}_f{num_channels}"
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
    data_path = data_cfg.get("dir_path", "./") + data_cfg.get("train_path", "")
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
