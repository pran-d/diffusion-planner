import argparse
import os
from typing import Any, Dict, Tuple

import yaml


DEFAULT_INFERENCE_CONFIG_PATH = "config/inference.yaml"


def load_inference_defaults(
    argv=None,
    section: str = "inference_mg",
    default_path: str = DEFAULT_INFERENCE_CONFIG_PATH,
) -> Tuple[Dict[str, Any], str]:
    """
    Load CLI default overrides from an inference YAML config.

    Precedence (lowest to highest):
      parser.add_argument(default=...) < YAML defaults < explicit CLI args

    YAML structure:
      common: {...}           # optional defaults shared across scripts
      <section>: {...}        # script-specific defaults

    Args:
        argv: CLI argv list (without program name). If None, uses process argv.
        section: Top-level YAML section to load for the caller.
        default_path: Fallback config path when --inference_config is not provided.

    Returns:
        (defaults_dict, resolved_path)
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--inference_config", type=str, default=default_path)
    known, _ = probe.parse_known_args(argv)

    path = known.inference_config
    if not os.path.exists(path):
        return {}, path

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        return {}, path

    defaults: Dict[str, Any] = {}
    common = raw.get("common", {})
    specific = raw.get(section, {})

    if isinstance(common, dict):
        defaults.update(common)
    if isinstance(specific, dict):
        defaults.update(specific)

    return defaults, path
