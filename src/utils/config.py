"""Configuration loading and path resolution."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml


def _find_project_root() -> Path:
    """Walk up from this file to find pyproject.toml."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml)")


PROJECT_ROOT = _find_project_root()


def load_yaml(name: str) -> Dict[str, Any]:
    """Load a YAML config from configs/ by name (e.g. 'paths' or 'sweep')."""
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class ProjectPaths:
    """Resolved absolute paths for all data locations."""

    project_root: Path

    # External datasets
    gsc_dir: Path
    polis_dir: Path
    remesh_dir: Path

    # Raw downloaded data
    habermas_dir: Path

    # Processed pipeline outputs
    eval_dir: Path
    issues_dir: Path
    opinions_dir: Path
    triplets_dir: Path

    # Models
    best_dir: Path

    # Results
    results_dir: Path

    @classmethod
    def from_yaml(cls, config_path: str = None) -> "ProjectPaths":
        root = PROJECT_ROOT
        if config_path:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = load_yaml("paths")

        flat = {}
        for section in cfg.values():
            if isinstance(section, dict):
                flat.update(section)

        resolved = {k: root / v for k, v in flat.items()}
        return cls(project_root=root, **resolved)

    @classmethod
    def auto(cls) -> "ProjectPaths":
        return cls.from_yaml()
