"""I/O, seeding, environment capture, logging."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

log = logging.getLogger("harf")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def open_text(path: str | os.PathLike, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | os.PathLike, obj, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, default=_json_default)


def read_json(path: str | os.PathLike):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash01(s: str, salt: str = "") -> float:
    """Deterministic hash in [0,1) — used for document-level splits (independent of PYTHONHASHSEED)."""
    h = hashlib.sha1((salt + s).encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def env_info() -> dict:
    pkgs = {}
    for p in ["torch", "transformers", "tokenizers", "peft", "accelerate", "datasets", "sacrebleu",
              "numpy", "scipy", "pandas"]:
        try:
            pkgs[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pkgs[p] = None
    info = {"python": sys.version, "platform": platform.platform(), "packages": pkgs, "git_commit": git_commit()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpus"] = [
                {"name": torch.cuda.get_device_name(i),
                 "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                for i in range(torch.cuda.device_count())]
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as e:
        raise SystemExit("This step needs PyTorch: pip install -e '.[gpu]'") from e
