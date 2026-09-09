"""Fail if tracked or unignored files expose common private research artifacts.

This is a targeted publication check, not a substitute for reviewing the files.
It also scans already-tracked files, which .gitignore alone cannot protect.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {"raw_sav", "data_dictionary", ".venv", "__pycache__", ".ssh"}
PRIVATE_SUFFIXES = {".sav", ".zsav", ".dta", ".por", ".tab", ".parquet", ".feather", ".arrow", ".rds", ".joblib", ".pkl", ".pickle", ".rtf", ".pem", ".key", ".zip"}
PRIVATE_NAMES = {".env", ".netrc", ".npmrc", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
PATTERNS = {
    "machine-specific path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "possible access credential": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"),
}


def main() -> int:
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        print("Run this check in an initialized Git checkout.", file=sys.stderr)
        return 2
    errors = []
    files = sorted(set(p for p in result.stdout.decode().split("\0") if p))
    for name in files:
        path = Path(name)
        if PRIVATE_DIRS.intersection(path.parts) or path.suffix.lower() in PRIVATE_SUFFIXES:
            errors.append(f"Private artifact: {name}")
        if path.name.lower() in PRIVATE_NAMES or path.name.lower().startswith(".env."):
            errors.append(f"Private configuration: {name}")
        if path.suffix.lower() == ".csv" and not (
            name.startswith("outputs/main/tables/") or name == "docs/correction-comparison.csv"
        ):
            errors.append(f"Unreviewed CSV outside aggregate results: {name}")
        if path.parts[0] == "outputs" and len(path.parts) > 1 and path.parts[1] not in {"main", "README.md"}:
            errors.append(f"Unreviewed run output: {name}")
        local = ROOT / path
        if not local.is_file() or path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg"}:
            continue
        content = local.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{label}: {name}")
        if path.suffix.lower() == ".csv" and "respondent_id" in content.partition("\n")[0].lower():
            errors.append(f"Respondent-level CSV: {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Publication check passed for {len(files)} candidate public files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
