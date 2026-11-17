"""Configuration loading.

Configs are YAML.  A config may ``extends: other.yaml`` (path relative to the
file).  A *run* config is the experiment config deep-merged with the chosen
method's overrides (``methods.<name>``) plus the run's target and seed.

The **science hash** of a run config excludes settings that only affect
reliability or file locations (retries, mirror dir, endpoints' URLs...).  A run
refuses to resume if its science hash changed, because that would silently mix
two different experiments in one trajectory.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .utils.io import short_hash
from .utils.runtime import ConfigError

# Keys (dotted) that never change results and therefore do not enter the science hash.
_NON_SCIENCE = [
    "reliability",
    "experiment.runs_dir",
    "experiment.data_dir",
    "experiment.hf_cache_dir",
    "controller.endpoints",
    "controller.retries",
    "controller.timeout_s",
    "controller.backoff_s",
    "controller.on_unavailable",
    "matrix",
    "methods",
    "benchmark",
    "logging",
]


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path, seen: set[Path] | None = None) -> dict:
    path = path.resolve()
    seen = seen or set()
    if path in seen:
        raise ConfigError(f"circular 'extends' at {path}")
    seen.add(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        base = _load_yaml((path.parent / parent), seen)
        data = deep_merge(base, data)
    return data


def parse_override(s: str) -> tuple[list[str], Any]:
    if "=" not in s:
        raise ConfigError(f"override '{s}' must look like a.b.c=value")
    k, v = s.split("=", 1)
    return k.strip().split("."), yaml.safe_load(v)


def set_dotted(d: dict, keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_dotted(d: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _del_dotted(d: dict, dotted: str) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return
        cur = cur[k]
    if isinstance(cur, dict):
        cur.pop(keys[-1], None)


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    cfg = _load_yaml(path)
    for o in overrides or []:
        keys, value = parse_override(o)
        set_dotted(cfg, keys, value)
    cfg.setdefault("experiment", {})
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_dir"] = str(path.resolve().parent)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for section in ("model", "pruning", "data", "search", "testing", "controller", "methods"):
        if section not in cfg:
            raise ConfigError(f"config is missing section '{section}'")
    mode = cfg["pruning"].get("mode")
    if mode not in ("unstructured", "nm", "structured"):
        raise ConfigError(f"pruning.mode must be unstructured|nm|structured, got {mode!r}")
    pol = cfg["testing"].get("policy")
    if pol not in ("none", "balanced", "loss_ranked", "neyman", "cap"):
        raise ConfigError(f"testing.policy invalid: {pol!r}")


def resolve_path(cfg: dict, p: str | None) -> Path | None:
    if p is None:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p)))
    pp = Path(p)
    return pp if pp.is_absolute() else (Path(cfg["_config_dir"]) / pp).resolve()


def run_config(cfg: dict, method: str, target: float, seed: int) -> dict:
    methods = cfg.get("methods", {})
    if method not in methods:
        raise ConfigError(f"unknown method '{method}'. Known: {', '.join(methods)}")
    m = methods[method] or {}
    rc = deep_merge({k: v for k, v in cfg.items() if k != "methods"}, m.get("overrides", {}))
    rc["run"] = {"method": method, "kind": m.get("kind", "search"), "target": float(target), "seed": int(seed)}
    rc["methods"] = {method: m}
    _validate({**rc, "methods": rc["methods"]})
    return rc


def science_hash(rc: dict) -> str:
    sci = copy.deepcopy({k: v for k, v in rc.items() if not k.startswith("_")})
    for k in _NON_SCIENCE:
        _del_dotted(sci, k)
    return short_hash(sci)


def run_id(method: str, target: float, seed: int) -> str:
    return f"{method}__t{int(round(target * 100)):02d}__s{seed}"


def runs_dir(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["experiment"].get("runs_dir", "runs"))


def data_dir(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["experiment"].get("data_dir", "data_cache"))


def loss_limit_final(rc: dict) -> float:
    """Final-sparsity domain-loss limit for this run's target (from a {target: limit} map or a scalar)."""
    spec = rc["testing"]["loss_limit"]["final"]
    target = rc["run"]["target"]
    if isinstance(spec, (int, float)):
        return float(spec)
    items = sorted((float(k), float(v)) for k, v in spec.items())
    # piecewise-linear interpolation on target; clamp at the ends
    if target <= items[0][0]:
        return items[0][1]
    if target >= items[-1][0]:
        return items[-1][1]
    for (t0, v0), (t1, v1) in zip(items, items[1:]):
        if t0 <= target <= t1:
            return v0 + (v1 - v0) * (target - t0) / (t1 - t0)
    return items[-1][1]
