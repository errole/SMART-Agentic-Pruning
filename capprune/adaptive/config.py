"""Loads an adaptive-evaluation config (YAML; supports ``extends`` and ``--set a.b=value`` like the main configs)."""
from __future__ import annotations

from pathlib import Path

from ..config import _load_yaml, parse_override, resolve_path, set_dotted
from ..utils.runtime import ConfigError


def load_adaptive(path: str | Path, overrides: list[str] | None = None) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    cfg = _load_yaml(p)
    for o in overrides or []:
        keys, value = parse_override(o)
        set_dotted(cfg, keys, value)
    for key in ("cache_dir", "out_dir", "evaluation"):  # (gate-design configs add a `design` section)
        if key not in cfg:
            raise ConfigError(f"{p}: missing '{key}'")
    cfg["_config_path"] = str(p.resolve())
    cfg["_config_dir"] = str(p.resolve().parent)
    return cfg


def cfg_path(cfg: dict, value: str) -> Path:
    return resolve_path(cfg, value)
