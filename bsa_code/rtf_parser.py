from __future__ import annotations

import re
from pathlib import Path


_HEX_ESCAPE_RE = re.compile(r"\\'[0-9a-fA-F]{2}")
_CONTROL_WORD_RE = re.compile(r"\\[a-zA-Z]+-?\d* ?")
_WHITESPACE_RE = re.compile(r"\s+")


def rtf_to_text(content: str) -> str:
    text = _HEX_ESCAPE_RE.sub(" ", content)
    text = _CONTROL_WORD_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_rtf_text(path: str | Path) -> str:
    return rtf_to_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def count_mentions(text: str, names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", flags=re.IGNORECASE)
        counts[name] = len(pattern.findall(text))
    return counts
