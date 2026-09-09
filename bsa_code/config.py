import json
from pathlib import Path


def _project_root(config_file: Path) -> Path:
    start = config_file.resolve()
    for folder in (start.parent, *start.parents):
        has_package = (folder / "bsa_code").is_dir()
        has_config = (folder / "config").is_dir()
        if has_package and has_config:
            return folder

        if (folder / "run_all.py").is_file() and (folder / "requirements.txt").is_file():
            return folder

    return start.parent


def _read_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def _as_project_path(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((root / path).resolve())


def _read_wave_files(config_file: Path, waves_dir: str) -> list[dict]:
    waves_path = (config_file.parent / waves_dir).resolve()
    if not waves_path.exists():
        raise FileNotFoundError(f"Waves directory not found: {waves_path}")
    if not waves_path.is_dir():
        raise NotADirectoryError(f"Waves path is not a directory: {waves_path}")

    waves = [_read_json_object(path) for path in sorted(waves_path.glob("*.json"))]
    try:
        return sorted(waves, key=lambda wave: int(wave.get("year")))
    except Exception as exc:
        raise ValueError(
            f"All wave config files in {waves_path} must contain an integer 'year'"
        ) from exc


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file(path: Path, seen: set[Path]) -> dict:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Config extends cycle detected at: {path}")

    config = _read_json_object(path)
    seen = seen | {resolved}

    waves_dir = config.pop("waves_dir", None)
    if waves_dir is not None:
        if not isinstance(waves_dir, str) or not waves_dir.strip():
            raise ValueError(f"Config 'waves_dir' must be a non-empty string: {path}")
        config["waves"] = _read_wave_files(path, waves_dir.strip())

    parent_config = config.pop("extends", None)
    if parent_config is None:
        return config
    if not isinstance(parent_config, str) or not parent_config.strip():
        raise ValueError(f"Config 'extends' must be a non-empty string: {path}")

    parent_path = (path.parent / parent_config).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Extended config not found: {parent_path}")

    return _deep_merge(_load_config_file(parent_path, seen), config)


def _resolve_path_at(config: dict, keys: tuple[str, ...], root: Path) -> None:
    section = config
    for key in keys[:-1]:
        section = section.get(key)
        if not isinstance(section, dict):
            return

    value = section.get(keys[-1])
    if isinstance(value, str) and value.strip():
        section[keys[-1]] = _as_project_path(root, value.strip())


def _resolve_paths(config: dict, root: Path) -> dict:
    config = dict(config)

    if isinstance(config.get("paths"), dict):
        paths = dict(config["paths"])
        for key, value in paths.items():
            if isinstance(value, str) and value.strip():
                paths[key] = _as_project_path(root, value.strip())
        config["paths"] = paths

    if isinstance(config.get("waves"), list):
        waves = []
        for wave in config["waves"]:
            if not isinstance(wave, dict):
                waves.append(wave)
                continue

            wave = dict(wave)
            if isinstance(wave.get("file"), str) and wave["file"].strip():
                wave["file"] = _as_project_path(root, wave["file"].strip())
            if isinstance(wave.get("dictionary"), str) and wave["dictionary"].strip():
                wave["dictionary"] = _as_project_path(root, wave["dictionary"].strip())
            waves.append(wave)
        config["waves"] = waves

    for keys in (
        ("harmonisation_qc", "summary_path"),
        ("measurement", "pca_structure", "path"),
    ):
        _resolve_path_at(config, keys, root)

    return config


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = _load_config_file(path, seen=set())
    return _resolve_paths(config, _project_root(path))


def ensure_dirs(paths: dict) -> None:
    for key in ("derived_dir", "outputs_dir"):
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
