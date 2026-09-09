# reads wave metadata, dictionaries, and attitude map
# writes the wave inventory table in the build output

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .attitude_map import load_attitude_map
from .config import load_config
from .logging_utils import log_event, setup_logging
from .meta_utils import read_wave_metadata, resolve_dictionary_path
from .rtf_parser import count_mentions, load_rtf_text


def build_inventory(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    derived_dir = Path(config["paths"]["derived_dir"])
    inventory_dir = derived_dir / "inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(derived_dir / "logs" / "inventory.log")

    attitude_map = load_attitude_map(config["paths"]["attitude_map"])
    rows: list[dict[str, object]] = []

    for wave in config["waves"]:
        year = int(wave["year"])
        metadata = read_wave_metadata(wave)
        columns = metadata["column_names"]
        dictionary_path = resolve_dictionary_path(config, wave)
        dictionary_exists = dictionary_path.exists()
        dictionary_text = load_rtf_text(dictionary_path) if dictionary_exists else ""

        source_vars = [str(wave["id_var"]), str(wave["weight_var"])]
        for spec in wave.get("demographics_map", {}).values():
            source = spec.get("source") if isinstance(spec, dict) else spec
            if source:
                source_vars.append(str(source))
        for construct in attitude_map.get("constructs", {}).values():
            for item in construct.get("items", {}).values():
                wave_spec = item.get("waves", {}).get(str(year))
                if wave_spec and wave_spec.get("var"):
                    source_vars.append(str(wave_spec["var"]))

        source_vars = list(dict.fromkeys(source_vars))
        mentions = count_mentions(dictionary_text, source_vars) if dictionary_text else {}
        present_sources = [name for name in source_vars if name in columns]

        rows.append(
            {
                "year": year,
                "file": str(wave["file"]),
                "file_type": str(wave["file_type"]),
                "rows": metadata["row_count"],
                "columns": int(len(columns)),
                "configured_sources": int(len(source_vars)),
                "configured_sources_present": int(len(present_sources)),
                "dictionary_path": str(dictionary_path),
                "dictionary_exists": bool(dictionary_exists),
                "dictionary_source_mentions": int(sum(mentions.values())),
            }
        )

        log_event(
            logger,
            "inventory_wave_complete",
            wave_year=year,
            rows=metadata["row_count"],
            columns=int(len(columns)),
            configured_sources=int(len(source_vars)),
            configured_sources_present=int(len(present_sources)),
            dictionary_exists=bool(dictionary_exists),
        )

    inventory_path = inventory_dir / "wave_inventory.csv"
    pd.DataFrame(rows).sort_values("year").to_csv(inventory_path, index=False)
    log_event(logger, "inventory_complete", output=str(inventory_path))
