import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np

_STYLE_TO_ID = {
    "pick": 0,
    "push": 1,
    "kick": 2,
}


def style_name_to_id(style: str) -> int:
    key = str(style).strip().lower().replace("_", "-")
    if key in ("pick", "pick-place", "pickplace"):
        return _STYLE_TO_ID["pick"]
    if key in ("push", "drag", "rotate"):
        return _STYLE_TO_ID["push"]
    if key in ("kick",):
        return _STYLE_TO_ID["kick"]
    raise ValueError(f"Unknown style: {style}")


def one_hot_style(style_id: int, num_styles: int = 3) -> np.ndarray:
    vec = np.zeros((num_styles,), dtype=np.float32)
    idx = int(style_id)
    if 0 <= idx < num_styles:
        vec[idx] = 1.0
    return vec


def _normalise_task_text(task_text: str) -> str:
    s = task_text.lower().replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def classify_style_from_task_text(task_text: str, rng: Optional[np.random.Generator] = None) -> Optional[int]:
    if not task_text:
        return None
    txt = _normalise_task_text(task_text)

    has_pick = any(k in txt for k in ["[pick]", " pick", "[hold]", "[place]", "carry"])
    has_kick = ("[kick] the box" == txt)
    has_push = any(k in txt for k in ["[push]", " push", "[drag]", " drag", "[rotate]", " rotate"])

    # User rule: mixed kick+pick is randomly assigned pick or kick.
    if has_pick and has_kick:
        _rng = rng if rng is not None else np.random.default_rng()
        return int(_rng.choice([_STYLE_TO_ID["pick"], _STYLE_TO_ID["kick"]]))

    if has_pick:
        return _STYLE_TO_ID["pick"]
    if has_kick:
        return _STYLE_TO_ID["kick"]
    if has_push:
        return _STYLE_TO_ID["push"]
    return None


def load_task_text_map(tasks_file: Optional[str]) -> Dict[str, str]:
    if not tasks_file:
        return {}
    if not os.path.exists(tasks_file):
        return {}

    mapping: Dict[str, str] = {}
    with open(tasks_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    parsed_any = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                folder = rec.get("original_folder") or rec.get("folder")
                task = rec.get("task")
                if folder and isinstance(task, str):
                    mapping[str(folder)] = task
                    parsed_any = True
        except Exception:
            continue

    if parsed_any:
        return mapping

    try:
        import yaml

        obj = yaml.safe_load("\n".join(lines))
        if isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    folder = rec.get("original_folder") or rec.get("folder")
                    task = rec.get("task")
                    if folder and isinstance(task, str):
                        mapping[str(folder)] = task
        elif isinstance(obj, dict):
            for folder, task in obj.items():
                if isinstance(task, str):
                    mapping[str(folder)] = task
    except Exception:
        pass

    return mapping


def classify_style_from_motion(
    base: np.ndarray,
    joints: np.ndarray,
    obj: np.ndarray,
    lift_threshold: float = 0.08,
    kick_joint_speed_threshold: float = 0.9,
    obj_speed_threshold: float = 0.12,
) -> Tuple[int, Dict[str, float]]:
    """Infer style from trajectory geometry only (heuristic)."""
    if base.ndim != 2 or joints.ndim != 2 or obj.ndim != 2:
        return _STYLE_TO_ID["push"], {"confidence": 0.0}

    obj_z0 = float(obj[0, 2])
    obj_lift = float(np.max(obj[:, 2]) - obj_z0)

    obj_xy = obj[:, :2]
    obj_xy_vel = np.linalg.norm(np.diff(obj_xy, axis=0, prepend=obj_xy[:1]), axis=-1)
    obj_peak_speed = float(np.max(obj_xy_vel))

    lower_joint_ids = [3, 4, 9, 10]
    lower_joint_ids = [j for j in lower_joint_ids if j < joints.shape[1]]
    if lower_joint_ids:
        lower_vel = np.abs(np.diff(joints[:, lower_joint_ids], axis=0, prepend=joints[:1, lower_joint_ids]))
        lower_peak = float(np.max(lower_vel))
    else:
        lower_peak = 0.0

    if obj_lift >= lift_threshold:
        return _STYLE_TO_ID["pick"], {
            "confidence": min(1.0, obj_lift / max(lift_threshold, 1e-6)),
            "obj_lift": obj_lift,
            "obj_peak_speed": obj_peak_speed,
            "lower_peak": lower_peak,
        }

    if (lower_peak >= kick_joint_speed_threshold) and (obj_peak_speed >= obj_speed_threshold):
        return _STYLE_TO_ID["kick"], {
            "confidence": min(
                1.0,
                0.5 * (lower_peak / max(kick_joint_speed_threshold, 1e-6))
                + 0.5 * (obj_peak_speed / max(obj_speed_threshold, 1e-6)),
            ),
            "obj_lift": obj_lift,
            "obj_peak_speed": obj_peak_speed,
            "lower_peak": lower_peak,
        }

    return _STYLE_TO_ID["push"], {
        "confidence": 0.5,
        "obj_lift": obj_lift,
        "obj_peak_speed": obj_peak_speed,
        "lower_peak": lower_peak,
    }
