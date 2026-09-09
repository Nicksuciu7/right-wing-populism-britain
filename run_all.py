"""Run the submitted analysis in an isolated, preflight-checked output directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from bsa_code.config import load_config

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_STAGE_COMMANDS = (
    "inventory", "ingest", "harmonise_demographics", "harmonise_attitudes",
    "score", "build_model_frame", "tune_model", "train_eval", "trends",
)


def validate_output_paths(config: dict, root: Path = REPO_ROOT) -> tuple[Path, Path]:
    """Reject output paths that could erase inputs, source, or each other."""
    output_root = (root / "outputs").resolve()
    targets = tuple(Path(config["paths"][key]).resolve() for key in ("derived_dir", "outputs_dir"))
    protected = [root / "raw_sav", root / "data_dictionary", root / "config", root / "bsa_code", root / "outputs" / "main"]
    protected.extend(Path(wave["file"]) for wave in config["waves"])
    protected.extend(Path(config["paths"][key]) for key in ("raw_dir", "dictionary_dir", "attitude_map"))
    for target in targets:
        if target == output_root or not target.is_relative_to(output_root):
            raise ValueError("Runner outputs must be separate directories below the project's outputs/ directory.")
        for source in protected:
            source = source.resolve()
            if target == source or source.is_relative_to(target) or target.is_relative_to(source):
                raise ValueError(f"Output overlaps protected input or source: {target}")
    if targets[0].is_relative_to(targets[1]) or targets[1].is_relative_to(targets[0]):
        raise ValueError("Build and presentation output directories must not overlap.")
    path = Path(config["harmonisation_qc"]["summary_path"]).resolve()
    if not path.is_relative_to(targets[0]) or path == targets[0]:
        raise ValueError("harmonisation_qc.summary_path must be a file below derived_dir.")
    structure = config.get("measurement", {}).get("pca_structure", {})
    if structure.get("mode", "fit") == "fit" and structure.get("path"):
        path = Path(structure["path"]).resolve()
        if not path.is_relative_to(targets[0]) or path == targets[0]:
            raise ValueError("Fitted PCA structure must be saved below derived_dir.")
    return targets


def _run(python: str, config: Path, stage: str) -> None:
    command = [python, "-m", "bsa_code", "--config", str(config), stage]
    print("+", " ".join(command), flush=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--config-template", dest="config", default="config/reproduce.json")
    parser.add_argument("--python-bin", help="Installed Python 3.11 or 3.12 environment; defaults to this interpreter.")
    parser.add_argument("--dry-run", action="store_true", help="Show execution order without reading survey data or writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the two configured output directories after successful input validation.")
    parser.add_argument("--no-validate-schema", action="store_true", help="Skip final output validation (not recommended).")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_config(config_path)
    targets = validate_output_paths(config)
    stages = list(PIPELINE_STAGE_COMMANDS)
    if not args.no_validate_schema:
        stages.append("validate_schema")
    python = args.python_bin or sys.executable
    if args.dry_run:
        print("Configuration:", config_path)
        print("Build:", targets[0])
        print("Results:", targets[1])
        print("Execution: validate_config -> " + " -> ".join(stages))
        return

    # Validate every input before touching an existing run.
    _run(python, config_path, "validate_config")
    occupied = [p for p in targets if p.exists() and (not p.is_dir() or any(p.iterdir()))]
    if occupied and not args.overwrite:
        parser.error("Output already exists. Choose a new configuration or use --overwrite to replace that run.")
    for path in targets:
        if path.exists() and args.overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        path.mkdir(parents=True, exist_ok=True)

    from bsa_code.provenance import write_run_manifest

    manifest_path = targets[0] / "run_manifest.json"
    write_run_manifest(manifest_path, config_path, config, status="running", root=REPO_ROOT, python=python)
    try:
        for stage in stages:
            _run(python, config_path, stage)
    except (subprocess.CalledProcessError, KeyboardInterrupt):
        manifest = json.loads(manifest_path.read_text())
        manifest.update(status="failed", failed_stage=stage)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status="complete", completed_stages=stages)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Complete. Results:", targets[1])


if __name__ == "__main__":
    main()
