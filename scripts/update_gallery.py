"""Copy updated pipeline figures only after verifying submitted aggregate results."""
from pathlib import Path
import shutil
from verify_results import compare_tables

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    compare_tables(ROOT / "outputs/main/tables", ROOT / "outputs/reproduced/main/tables")
    originals = ROOT / "outputs/main/figures"
    reproduced = ROOT / "outputs/reproduced/main/figures"
    destination = ROOT / "docs/figures/analysis"
    paths = [p.relative_to(originals) for p in sorted(originals.rglob("*.png"))]
    missing = [str(p) for p in paths if not (reproduced / p).is_file()]
    if missing:
        raise ValueError(f"Missing regenerated figures: {missing}")
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reproduced / relative, target)
    print(f"Published {len(paths)} regenerated figures to the local documentation gallery.")


if __name__ == "__main__":
    main()
