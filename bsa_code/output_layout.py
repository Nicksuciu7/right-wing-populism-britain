from __future__ import annotations

from pathlib import Path


def measurement_tables_dir(main_dir: Path) -> Path:
    return Path(main_dir) / "tables" / "measurement"


def measurement_figures_dir(main_dir: Path) -> Path:
    return Path(main_dir) / "figures" / "measurement"


def modelling_tables_dir(main_dir: Path) -> Path:
    return Path(main_dir) / "tables" / "modelling"


def modelling_figures_dir(main_dir: Path) -> Path:
    return Path(main_dir) / "figures" / "modelling"


def trend_tables_dir(main_dir: Path) -> Path:
    return Path(main_dir) / "tables" / "trends"


def trend_figures_dir(main_dir: Path) -> Path:
    return Path(main_dir) / "figures" / "trends"


def build_diagnostics_dir(build_dir: Path) -> Path:
    return Path(build_dir) / "diagnostics"


def build_model_frame_dir(build_dir: Path) -> Path:
    return Path(build_dir) / "model_frame"


def build_model_frame_parquet_path(build_dir: Path) -> Path:
    return build_model_frame_dir(build_dir) / "model_frame.parquet"


def build_model_frame_meta_path(build_dir: Path) -> Path:
    return build_model_frame_dir(build_dir) / "model_frame_meta.json"
