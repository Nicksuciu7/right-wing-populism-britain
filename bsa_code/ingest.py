# ingests raw survey files into parquet files

from __future__ import annotations

import json
from pathlib import Path

from .attitude_map import load_attitude_map
from .config import load_config
from .logging_utils import log_event, setup_logging
from .meta_utils import read_wave_frame


def ingest_raw_waves(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    derived_dir = Path(config["paths"]["derived_dir"])
    raw_dir = derived_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(derived_dir / "logs" / "ingest.log")

    attitude_map = load_attitude_map(config["paths"]["attitude_map"])

    for wave in config["waves"]:
        year = int(wave["year"])
        columns = [str(wave["id_var"]), str(wave["weight_var"])]

        for spec in wave.get("demographics_map", {}).values():
            source = spec.get("source") if isinstance(spec, dict) else spec
            if source:
                columns.append(str(source))

        for construct in attitude_map.get("constructs", {}).values():
            for item in construct.get("items", {}).values():
                wave_spec = item.get("waves", {}).get(str(year))
                if wave_spec and wave_spec.get("var"):
                    columns.append(str(wave_spec["var"]))

        columns = list(dict.fromkeys(columns))
        frame = read_wave_frame(wave, usecols=columns)

        raw_path = raw_dir / f"wave_{year}.parquet"
        meta_path = raw_dir / f"wave_{year}_meta.json"
        frame.to_parquet(raw_path, index=False)
        meta_path.write_text(
            json.dumps(
                {
                    "year": year,
                    "rows": int(len(frame)),
                    "columns": columns,
                    "id_var": str(wave["id_var"]),
                    "weight_var": str(wave["weight_var"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        log_event(
            logger,
            "ingest_wave_complete",
            wave_year=year,
            rows=int(len(frame)),
            columns=int(len(columns)),
            output=str(raw_path),
        )

    log_event(logger, "ingest_complete", output=str(raw_dir))
