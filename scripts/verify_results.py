"""Compare a submitted-method rerun against the frozen aggregate CSVs."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare_tables(reference: Path, candidate: Path) -> list[str]:
    checked = []
    for original in sorted(reference.rglob("*.csv")):
        relative = original.relative_to(reference)
        reproduced = candidate / relative
        if not reproduced.is_file():
            raise AssertionError(f"Missing reproduced table: {relative}")
        pd.testing.assert_frame_equal(pd.read_csv(original), pd.read_csv(reproduced),
                                      check_exact=False, rtol=1e-10, atol=1e-12,
                                      obj=str(relative))
        checked.append(str(relative))
    if not checked:
        raise AssertionError("Reference directory contains no tables.")
    extras = {str(p.relative_to(candidate)) for p in candidate.rglob("*.csv")} - set(checked)
    if extras:
        raise AssertionError(f"Unexpected reproduced tables: {sorted(extras)}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/main/tables")
    parser.add_argument("--candidate", type=Path, default=ROOT / "outputs/reproduced/main/tables")
    args = parser.parse_args()
    checked = compare_tables(args.reference, args.candidate)
    print(f"All {len(checked)} aggregate tables match (rtol=1e-10, atol=1e-12).")


if __name__ == "__main__":
    main()
