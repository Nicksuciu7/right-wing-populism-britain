# Reproducing the project

## Choose the appropriate route

| Route | Requires licensed BSA files? | Command | Destination |
|---|---|---|---|
| Read and regenerate the portfolio figures | No | `python -m bsa_code.showcase` | `docs/figures/` |
| Run numerical and engineering tests | No | `python -m unittest discover -s tests -v` | Console |
| Preview the execution order | No | `python run_all.py --dry-run` | Console |
| Reproduce the submitted methodology | Yes | `python run_all.py` | `outputs/reproduced/` |
| Evaluate the documented corrections | Yes | `python run_all.py --config config/corrected.json` | `outputs/corrected/` |

`outputs/main/` is the frozen submission reference. Never report corrected outputs as though they were in the submitted dissertation.

## Installation

From a source checkout with Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

If reusing the original dissertation environment, first run `python -m pip uninstall diss-code` before reinstalling this package. The old distribution installed a separate `bsa_code` copy that could shadow the editable checkout outside the repository. A fresh environment does not have this conflict.

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell. The remaining `python` commands are the same. On macOS/Linux, the figure backend defaults to `Agg` under the runner, so no display server is needed.

`pyproject.toml` declares direct requirements; `requirements-lock.txt` captures the transitive analysis environment. The reference environment was Python 3.11.14 on macOS arm64. Floating-point library and platform differences can affect final digits. The lock is not a cryptographic dependency lock, and this work does not claim to have executed the GitHub Actions matrix remotely.

The source checkout is required: configuration, saved tables and the runner are repository assets, not bundled datasets in a pip wheel. The console command `bsa-rwp` is equivalent to `python -m bsa_code`.

## Data and input validation

Obtain the full five-wave datasets under your own access permissions, following [the data guide](../data/README.md). Run:

```bash
python -m bsa_code --config config/reproduce.json validate_config
```

Validation checks configured years, source files and columns, usable identifiers, duplicate identifiers and survey weights. The model-frame stage enforces one-to-one join keys. A newer archive edition may have different variables; investigate the documentation rather than bypassing the checks.

No survey data are downloaded by the pipeline. An error naming a missing `.sav` file means that the required local input is absent. The `--dry-run` route deliberately works without those files.

## Execution order and intermediate products

| Stage | Main responsibility | Local build products |
|---|---|---|
| `inventory` | Inspect survey metadata and mapped variable availability | `inventory/` |
| `ingest` | Read required columns; preserve metadata | `raw/` Parquet and JSON |
| `harmonise_demographics` | Align demographic categories and age codes | `demographics/`, harmonisation diagnostics |
| `harmonise_attitudes` | Remove invalid codes; rescale/reverse responses | `attitudes/`, harmonisation diagnostics |
| `score` | Fit/reuse measurement structure; score; derive fixed threshold | `scores/`, `labels/`, `models/`, measurement diagnostics |
| `build_model_frame` | One-to-one join and predictor coverage checks | `model_frame/` |
| `tune_model` | Five-fold training-only ridge CV | Tuning decision JSON and aggregate grid |
| `train_eval` | Baselines, ridge, internal and held-out-wave evaluation | Model object, metadata and aggregate metrics |
| `trends` | Descriptive wave models, permutation importance and subgroup summaries | Aggregate trend tables and figures |
| `validate_schema` | Verify expected products, columns and dtypes | Raises on failure |

The runner performs `validate_config` before any output replacement. It then records input/source/configuration hashes, environment versions and completion status in `outputs/reproduced/build/run_manifest.json`. Detailed logs remain private in `build/logs/`.

To repeat a run, use `python run_all.py --overwrite`. This explicitly replaces only the two configured output directories after path and input validation. The path guard rejects root directories, overlapping outputs, source/input locations and resolved symlinks outside the permitted output tree. It does not treat interrupted partial outputs as a completed run.

For an individual stage, keep the configuration consistent:

```bash
python -m bsa_code --config config/reproduce.json tune_model
python -m bsa_code --config config/reproduce.json train_eval
python -m bsa_code --config config/reproduce.json validate_schema
```

Individual stages overwrite their own managed files. The full runner provides the stronger rerun guard. Tuning writes its decision as a build artifact; it no longer changes the input JSON. Training consumes the selected alpha only when configuration and model-frame hashes match the tuning record. A stale decision requires rerunning `tune_model`. With no tuning record, a direct training stage uses the explicitly configured alpha.

## Check the numerical reproduction

```bash
python scripts/verify_results.py
```

This compares all 11 public CSVs, including item assignments, tuning grid, model metrics and trend outputs. It checks both values and schemas with `rtol=1e-10`, `atol=1e-12`. It also rejects missing or extra tables. Figure pixels are not compared because typography, resolution and axes were intentionally improved.

The supplied five-wave files reproduced every table. Original figures and tables remain byte-for-byte preserved in `outputs/main/`; [the validation record](validation.json) records the completed checks. The corrected profile is expected to differ and should not be passed to this submitted-results comparison as if equality were its goal.

## Optional discovery and report preparation

Dictionary discovery is a lexicon/phrase search that proposes candidates; it does not replace the curated item maps or train a language model:

```bash
python -m bsa_code --config config/reproduce.json nlp_discovery
```

Supply the five RTF dictionaries described in the data guide. They are ignored by Git. The seed lexicon now resolves relative to the project configuration rather than the current shell directory.

The public dissertation can be recreated from the original 39-page submission with the optional PDF dependency:

```bash
python -m pip install '.[report]'
python scripts/make_public_dissertation.py /path/to/submitted-dissertation.pdf
```

The script is specific to this submission, fails if the expected page count/boundaries are absent, and never overwrites its input. Its output omits the candidate identifier, proposal and signed forms. The public copy is already included; this dependency is unnecessary for the research pipeline.

## Public repository checks

```bash
python scripts/check_public_files.py
```

This checks both already-tracked and unignored files for restricted formats/directories, respondent-level CSV headers, common credential patterns and machine-specific paths. It is a focused audit aid, not a guarantee that every possible secret or disclosure is detectable. Review any newly added artifacts before publication; `.gitignore` does not remove a file from existing Git history.

GitHub Actions runs the data-free tests, publication check, aggregate figure renderer and dry-run on Python 3.11/3.12. Licensed survey data and a full private-data pipeline run are not part of public CI.
