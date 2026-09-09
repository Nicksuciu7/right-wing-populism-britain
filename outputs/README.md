# Outputs and provenance

- `main/`: original submitted aggregate tables and figures. Kept unchanged as the numerical and visual reference for the dissertation.
- `build/`: original local intermediate data and model artifacts; ignored by Git.
- `reproduced/`: isolated rerun of the submitted methodology, including updated plot formatting; ignored by Git.
- `corrected/`: separately labelled rerun with the two documented demographic corrections; ignored by Git.

Public presentation figures are in `docs/figures/`. They are regenerated from the submitted aggregate CSVs with `python -m bsa_code.showcase`; no licensed survey files are required. They are presentation updates, not new estimates.

Every complete run writes a private `build/run_manifest.json` containing input, source and configuration hashes, installed package versions and stage status. A failed run is marked as failed, not represented as a completed reproduction. `docs/validation.json` records the checks actually performed during the repository audit.
