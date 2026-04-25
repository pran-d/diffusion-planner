import copy
from typing import Dict, List, Tuple

from utils.math.sbto_utils import build_feature_layout

DEFAULT_PHASE1_FEATURES = [
    "delta_xy",
    "delta_yaw",
    "body_z",
    "obj_rel_pos",
    "obj_rel_rot6d",
]

DEFAULT_PHASE2_FEATURES = [
    "joints",
    "body_rot6d",
]


def _filter_valid_features(features: List[str]) -> List[str]:
    layout = build_feature_layout()
    return [k for k in features if k in layout and layout[k] > 0]


def resolve_two_phase_cfg(raw_cfg: Dict) -> Dict:
    cfg = copy.deepcopy(raw_cfg or {})
    phase1 = _filter_valid_features(cfg.get("phase1_features", DEFAULT_PHASE1_FEATURES))
    phase2 = _filter_valid_features(cfg.get("phase2_features", DEFAULT_PHASE2_FEATURES))
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "phase1_features": phase1,
        "phase2_features": phase2,
        "phase1_checkpoint": cfg.get("phase1_checkpoint", None),
        "phase2_checkpoint": cfg.get("phase2_checkpoint", None),
    }


def feature_dims(feature_order: List[str]) -> int:
    layout = build_feature_layout()
    return int(sum(layout[k] for k in feature_order))


def build_feature_slices(feature_order: List[str]) -> Dict[str, slice]:
    layout = build_feature_layout()
    out = {}
    idx = 0
    for key in feature_order:
        dim = int(layout.get(key, 0))
        if dim <= 0:
            continue
        out[key] = slice(idx, idx + dim)
        idx += dim
    return out


def apply_phase_to_configs(
    model_cfg: Dict,
    data_cfg: Dict,
    training_cfg: Dict,
    noise_cfg: Dict,
    phase: str,
    two_phase_cfg: Dict,
) -> Tuple[Dict, Dict, Dict, Dict]:
    if phase not in ("phase1", "phase2"):
        raise ValueError(f"Unknown phase={phase!r}")

    out_model = copy.deepcopy(model_cfg)
    out_data = copy.deepcopy(data_cfg)
    out_train = copy.deepcopy(training_cfg)
    out_noise = copy.deepcopy(noise_cfg)

    phase_features = (
        two_phase_cfg["phase1_features"] if phase == "phase1" else two_phase_cfg["phase2_features"]
    )

    full_feature_order = list(data_cfg.get("feature_order", []))
    full_num_features = int(data_cfg.get("num_features", feature_dims(full_feature_order)))
    full_num_obs = int(data_cfg.get("num_observations", full_num_features))
    full_obs_start = max(0, full_num_features - full_num_obs)
    full_slices = build_feature_slices(full_feature_order)

    out_data["feature_order"] = list(phase_features)
    out_data["num_features"] = feature_dims(phase_features)
    obs_dim = 0
    for k in phase_features:
        sl = full_slices.get(k)
        if sl is None:
            continue
        inter_start = max(sl.start, full_obs_start)
        inter_stop = min(sl.stop, full_num_features)
        if inter_stop > inter_start:
            obs_dim += inter_stop - inter_start
    out_data["num_observations"] = int(max(0, min(obs_dim, out_data["num_features"])))
    out_data["two_phase_enabled"] = True
    out_data["two_phase_phase"] = phase
    out_data["phase1_context_features"] = list(two_phase_cfg["phase1_features"])
    out_data["phase1_context_mode"] = "last"
    
    out_data["full_feature_order"] = full_feature_order
    out_data["full_num_features"] = full_num_features
    out_data["full_num_observations"] = full_num_obs

    phase1_ctx_dim = feature_dims(two_phase_cfg["phase1_features"])
    out_data["phase1_context_dim"] = int(phase1_ctx_dim)
    out_model["phase1_condition"] = bool(phase == "phase2")
    out_data["phase1_condition"] = bool(phase == "phase2")
    out_model["phase1_context_dim"] = int(phase1_ctx_dim)
    
    if phase == "phase2":
        out_model["state_condition"] = True
        out_model["task_condition"] = False
        out_model["num_observations"] = full_num_obs
        
        out_data["state_condition"] = True
        out_data["task_condition"] = False
        out_data["num_observations"] = full_num_obs

    old_suffix = str(out_train.get("suffix", "")).strip()
    phase_suffix = f"{phase}"
    out_train["suffix"] = f"{old_suffix}_{phase_suffix}" if old_suffix else f"_{phase_suffix}"

    out_noise.setdefault("hierarchical_noise", {})
    out_noise["hierarchical_noise"]["enabled"] = False
    out_noise["hierarchical_noise"]["phase1_feature_keys"] = list(two_phase_cfg["phase1_features"])
    out_noise["hierarchical_noise"]["phase2_feature_keys"] = list(two_phase_cfg["phase2_features"])

    return out_model, out_data, out_train, out_noise
