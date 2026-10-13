"""Config loading: YAML with `extends`, CLI `--set a.b=value` overrides, condition merging."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str | os.PathLike) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_update(_load_yaml(path.parent / parent), cfg)
    return cfg


def parse_overrides(items: list[str] | None) -> dict:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def load_config(path: str, overrides: list[str] | None = None, lang: str | None = None,
                condition: str | None = None) -> dict:
    cfg = _load_yaml(path)
    cfg = deep_update(cfg, parse_overrides(overrides))
    if lang:
        cfg["lang"] = lang
    if condition:
        if condition not in cfg.get("conditions", {}):
            raise KeyError(f"Unknown condition {condition!r}; known: {list(cfg.get('conditions', {}))}")
        cfg["condition"] = condition
        cfg["cond"] = dict(cfg["conditions"][condition])
        # per-condition settings, e.g.  C_arabic: {..., overrides: {projection: {pairs: arabic}}}
        cfg = deep_update(cfg, cfg["cond"].get("overrides", {}))
    cfg["_config_path"] = str(path)
    return cfg


def langs_of(cfg: dict) -> list[str]:
    return cfg["lang"].split("+")


def model_slug(cfg: dict) -> str:
    return cfg["model"]["name"].split("/")[-1]


def lang_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["runs"]) / model_slug(cfg) / cfg["lang"]


def shared_dir(cfg: dict) -> Path:
    """Artifacts shared by all conditions of one (model, lang): baseline audit, merge list, projection."""
    d = lang_dir(cfg) / "_shared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cond_dir(cfg: dict, condition: str | None = None) -> Path:
    condition = condition or cfg.get("condition")
    if not condition:
        raise ValueError("No condition given (use --condition).")
    d = lang_dir(cfg) / condition
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_resolved(cfg: dict, out_dir: Path, name: str = "config.resolved.yaml") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
