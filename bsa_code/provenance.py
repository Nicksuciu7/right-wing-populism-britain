"""Reproducibility records; manifests stay with private build artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def read_tuned_alpha(config: dict, decision_path: Path, model_frame_path: Path) -> float | None:
    """Use a CV decision only for the exact configuration and model frame it saw."""
    if not decision_path.exists():
        return None
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    expected = {"configuration": config_sha256(config), "model_frame": file_sha256(model_frame_path)}
    if decision.get("input_fingerprint") != expected:
        raise ValueError("Tuning decision is stale or lacks provenance. Rerun tune_model for this configuration before train_eval.")
    alpha = float(decision["cv"]["selected"]["alpha"])
    if not 0 < alpha < float("inf"):
        raise ValueError("Tuned alpha must be finite and positive.")
    return alpha


def write_run_manifest(path: Path, config_path: Path, config: dict, *, status: str, root: Path, python: str) -> None:
    """Record input hashes and the actual stage interpreter without copying rows."""
    environment = subprocess.run(
        [python, "-c", "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version.split()[0], 'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"],
        check=True, capture_output=True, text=True,
    )
    def relative(p: Path) -> str:
        p = p.resolve()
        return str(p.relative_to(root)) if p.is_relative_to(root) else p.name
    manifest = {
        "status": status,
        "configuration": relative(config_path),
        "resolved_configuration_sha256": config_sha256(config),
        "environment": json.loads(environment.stdout),
        "input_sha256": {relative(Path(w["file"])): file_sha256(Path(w["file"])) for w in config["waves"]},
        "source_sha256": {relative(p): file_sha256(p) for p in sorted([*(root / "bsa_code").glob("*.py"), root / "run_all.py", root / "pyproject.toml", root / "requirements-lock.txt"])},
        "config_sha256": {relative(p): file_sha256(p) for p in sorted((root / "config").rglob("*.json")) if not p.name.startswith(".")},
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
