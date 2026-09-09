from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat


def read_wave_frame(wave: dict, *, usecols: list[str] | None = None) -> pd.DataFrame:
    file_path = Path(wave["file"])
    file_type = str(wave["file_type"]).strip().lower()
    if file_type == "sav":
        return pyreadstat.read_sav(
            file_path,
            usecols=usecols,
            apply_value_formats=False,
            formats_as_category=False,
            user_missing=False,
        )[0]
    if file_type == "tab":
        return pd.read_csv(file_path, sep="\t", usecols=usecols, low_memory=False)
    raise ValueError(f"Unsupported wave file_type: {file_type}")


def read_wave_metadata(wave: dict) -> dict[str, object]:
    file_path = Path(wave["file"])
    file_type = str(wave["file_type"]).strip().lower()
    if file_type == "sav":
        meta = pyreadstat.read_sav(file_path, metadataonly=True)[1]
        row_count = int(meta.number_rows) if meta.number_rows is not None else None
        return {
            "row_count": row_count,
            "column_names": list(meta.column_names or []),
        }
    if file_type == "tab":
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().strip("\n")
        columns = header.split("\t") if header else []
        row_count = None
        return {
            "row_count": row_count,
            "column_names": columns,
        }
    raise ValueError(f"Unsupported wave file_type: {file_type}")


def resolve_dictionary_path(config: dict, wave: dict) -> Path:
    if wave.get("dictionary"):
        return Path(wave["dictionary"])

    dictionary_dir = Path(config["paths"]["dictionary_dir"])
    year = int(wave["year"])
    suffix = str(year)[-2:]
    return dictionary_dir / f"bsa{suffix}_data_dictionary.rtf"
