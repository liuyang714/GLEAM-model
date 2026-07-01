"""YAML config loader"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load config.yaml.

    Search order:
      1. Explicit path
      2. Environment variable RENAL_CLS_CONFIG
      3. config.yaml in project root
    """
    if config_path is None:
        env_path = os.environ.get("RENAL_CLS_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_device(cfg: dict) -> "torch.device":
    """Return torch.device based on config"""
    import torch

    dev_str = cfg.get("common", {}).get("cuda_device", "0")
    if dev_str.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{dev_str}")
