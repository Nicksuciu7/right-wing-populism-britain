# loads and merges the attitude json files
# used by harmonisation and scoring stages

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Attitude map payload must be a JSON object: {path}")
    return payload


def _initialise_base_structure(base_payload: dict) -> dict:
    merged = {"constructs": {}}
    for construct, construct_payload in base_payload.get("constructs", {}).items():
        construct_payload = construct_payload if isinstance(construct_payload, dict) else {}
        merged_construct = {
            key: value
            for key, value in construct_payload.items()
            if key != "items"
        }
        merged_items = {}
        for item, item_payload in (construct_payload.get("items", {}) or {}).items():
            item_payload = item_payload if isinstance(item_payload, dict) else {}
            merged_item = {key: value for key, value in item_payload.items() if key != "waves"}
            merged_item["waves"] = {}
            merged_items[item] = merged_item
        merged_construct["items"] = merged_items
        merged["constructs"][construct] = merged_construct
    return merged


def _merge_year_payload(merged: dict, year_payload: dict, source_path: Path) -> None:
    year = year_payload.get("year")
    if year is None:
        raise ValueError(f"Attitude year payload missing 'year': {source_path}")
    try:
        year_key = str(int(year))
    except Exception as exc:
        raise ValueError(f"Attitude year payload has invalid 'year': {source_path}") from exc

    constructs = year_payload.get("constructs", {})
    if not isinstance(constructs, dict):
        raise ValueError(f"Attitude year payload constructs must be an object: {source_path}")

    for construct, construct_payload in constructs.items():
        construct_payload = construct_payload if isinstance(construct_payload, dict) else {}
        merged_construct = merged["constructs"].setdefault(construct, {"items": {}})
        if "description" in construct_payload and "description" not in merged_construct:
            merged_construct["description"] = construct_payload["description"]

        items = construct_payload.get("items", {})
        if not isinstance(items, dict):
            raise ValueError(f"Attitude year payload items must be an object: {source_path}")

        for item, spec in items.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Attitude year payload item spec must be an object: {source_path}::{construct}.{item}"
                )
            merged_item = merged_construct["items"].setdefault(item, {"waves": {}})
            waves = merged_item.setdefault("waves", {})
            if year_key in waves:
                raise ValueError(
                    f"Duplicate attitude mapping for year {year_key}: {construct}.{item} in {source_path}"
                )
            waves[year_key] = spec


def _load_attitude_map_dir(path: Path) -> dict:
    base_path = path / "base.json"
    if not base_path.exists():
        raise FileNotFoundError(f"Attitude map base file not found: {base_path}")

    base_payload = _load_json(base_path)
    merged = _initialise_base_structure(base_payload)

    years_dir = path / "years"
    if years_dir.exists():
        if not years_dir.is_dir():
            raise NotADirectoryError(f"Attitude map years path is not a directory: {years_dir}")
        for year_path in sorted(years_dir.glob("*.json")):
            year_payload = _load_json(year_path)
            _merge_year_payload(merged, year_payload, year_path)

    return merged


def load_attitude_map(path: str | Path) -> dict:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Attitude map path not found: {resolved}")
    if resolved.is_dir():
        return _load_attitude_map_dir(resolved)
    return _load_json(resolved)
