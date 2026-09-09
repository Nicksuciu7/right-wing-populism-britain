import json
import logging
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(log_path: str | Path) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bsa_code")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def log_event(logger: logging.Logger, stage: str, wave_year: int | None = None, **kwargs) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "stage": stage,
    }
    if wave_year is not None:
        payload["wave_year"] = wave_year
    payload.update(kwargs)
    logger.info(json.dumps(payload))
