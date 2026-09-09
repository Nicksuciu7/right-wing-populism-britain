# harmonises attitude items for each wave
# reads raw files and writes attitude parquets

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .attitude_map import load_attitude_map
from .config import load_config
from .harmonisation_utils import (
    build_respondent_ids,
    clean_weight,
    make_summary_row,
    rescale_attitude,
    write_stage_summary,
)
from .logging_utils import log_event, setup_logging


def harmonise_attitudes(config_path: str = "config/code_config.json") -> None:
    config = load_config(config_path)
    derived_dir = Path(config["paths"]["derived_dir"])
    raw_dir = derived_dir / "raw"
    output_dir = derived_dir / "attitudes"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(derived_dir / "logs" / "attitudes.log")

    attitude_map = load_attitude_map(config["paths"]["attitude_map"])
    summary_rows: list[dict[str, object]] = []

    for wave in config["waves"]:
        year = int(wave["year"])
        frame = pd.read_parquet(raw_dir / f"wave_{year}.parquet")
        missing_codes = list(wave.get("missing_codes", []))

        result = pd.DataFrame(
            {
                "respondent_id": build_respondent_ids(frame[wave["id_var"]], year),
                "wave_year": int(year),
                "weight": clean_weight(frame[wave["weight_var"]], missing_codes),
            }
        )

        for construct_name, construct in attitude_map.get("constructs", {}).items():
            for item_name, item in construct.get("items", {}).items():
                spec = item.get("waves", {}).get(str(year))
                if not spec:
                    continue
                output_name = f"{construct_name}__{item_name}"
                values = rescale_attitude(frame[spec["var"]], spec=spec, missing_codes=missing_codes)
                result[output_name] = values
                summary_rows.append(
                    make_summary_row(
                        stage="attitudes",
                        year=year,
                        output_name=output_name,
                        source_var=str(spec["var"]),
                        values=values,
                    )
                )

        output_path = output_dir / f"attitudes_{year}.parquet"
        result.to_parquet(output_path, index=False)
        log_event(
            logger,
            "attitudes_wave_complete",
            wave_year=year,
            rows=int(len(result)),
            columns=int(len(result.columns) - 3),
            output=str(output_path),
        )

    write_stage_summary(config["harmonisation_qc"]["summary_path"], "attitudes", summary_rows)
    log_event(logger, "attitudes_complete", output=str(output_dir))
