# File-by-file change record

Compared against the pre-edit local snapshot. The input folder had no Git history. Raw survey data, dictionaries, original intermediate data and the existing environment were retained locally and excluded from public file lists.

## Substantially modified original files

- [`bsa_code/__main__.py`](../bsa_code/__main__.py)
- [`bsa_code/config.py`](../bsa_code/config.py)
- [`bsa_code/demographics.py`](../bsa_code/demographics.py)
- [`bsa_code/harmonisation_utils.py`](../bsa_code/harmonisation_utils.py)
- [`bsa_code/logging_utils.py`](../bsa_code/logging_utils.py)
- [`bsa_code/model_tuning.py`](../bsa_code/model_tuning.py)
- [`bsa_code/nlp_item_discovery.py`](../bsa_code/nlp_item_discovery.py)
- [`bsa_code/plotting.py`](../bsa_code/plotting.py)
- [`bsa_code/scoring.py`](../bsa_code/scoring.py)
- [`bsa_code/train_eval.py`](../bsa_code/train_eval.py)
- [`bsa_code/trends.py`](../bsa_code/trends.py)
- [`pyproject.toml`](../pyproject.toml)
- [`run_all.py`](../run_all.py)

## Created public files

- [`.gitattributes`](../.gitattributes)
- [`.github/workflows/checks.yml`](../.github/workflows/checks.yml)
- [`.gitignore`](../.gitignore)
- [`CITATION.cff`](../CITATION.cff)
- [`LICENSE`](../LICENSE)
- [`README.md`](../README.md)
- [`bsa_code/provenance.py`](../bsa_code/provenance.py)
- [`bsa_code/showcase.py`](../bsa_code/showcase.py)
- [`config/corrected.json`](../config/corrected.json)
- [`config/reproduce.json`](../config/reproduce.json)
- [`data/README.md`](../data/README.md)
- [`dissertation/README.md`](../dissertation/README.md)
- [`dissertation/dissertation-public.pdf`](../dissertation/dissertation-public.pdf)
- [`docs/audit.md`](../docs/audit.md)
- [`docs/changes.md`](../docs/changes.md)
- [`docs/correction-comparison.csv`](../docs/correction-comparison.csv)
- [`docs/figures.md`](../docs/figures.md)
- [`docs/figures/analysis/measurement/extras/pca_correlation_matrix_right.png`](../docs/figures/analysis/measurement/extras/pca_correlation_matrix_right.png)
- [`docs/figures/analysis/measurement/extras/pca_correlation_matrix_welfare.png`](../docs/figures/analysis/measurement/extras/pca_correlation_matrix_welfare.png)
- [`docs/figures/analysis/measurement/measurement_component_questions_table.png`](../docs/figures/analysis/measurement/measurement_component_questions_table.png)
- [`docs/figures/analysis/measurement/measurement_component_summary_table.png`](../docs/figures/analysis/measurement/measurement_component_summary_table.png)
- [`docs/figures/analysis/measurement/pca_correlation_matrix_rwp.png`](../docs/figures/analysis/measurement/pca_correlation_matrix_rwp.png)
- [`docs/figures/analysis/measurement/pca_scree_rwp.png`](../docs/figures/analysis/measurement/pca_scree_rwp.png)
- [`docs/figures/analysis/modelling/predicted_vs_actual.png`](../docs/figures/analysis/modelling/predicted_vs_actual.png)
- [`docs/figures/analysis/modelling/regression_metrics_table.png`](../docs/figures/analysis/modelling/regression_metrics_table.png)
- [`docs/figures/analysis/trends/education_by_year_rwp.png`](../docs/figures/analysis/trends/education_by_year_rwp.png)
- [`docs/figures/analysis/trends/employment_status_by_year_rwp.png`](../docs/figures/analysis/trends/employment_status_by_year_rwp.png)
- [`docs/figures/analysis/trends/predictor_trend_model_fit_table.png`](../docs/figures/analysis/trends/predictor_trend_model_fit_table.png)
- [`docs/figures/analysis/trends/predictor_trends.png`](../docs/figures/analysis/trends/predictor_trends.png)
- [`docs/figures/analysis/trends/religion_by_year_rwp.png`](../docs/figures/analysis/trends/religion_by_year_rwp.png)
- [`docs/figures/analysis/trends/rwp_prevalence_by_year.png`](../docs/figures/analysis/trends/rwp_prevalence_by_year.png)
- [`docs/figures/model-performance.png`](../docs/figures/model-performance.png)
- [`docs/figures/predictor-importance.png`](../docs/figures/predictor-importance.png)
- [`docs/figures/prevalence.png`](../docs/figures/prevalence.png)
- [`docs/methodology.md`](../docs/methodology.md)
- [`docs/reproducibility.md`](../docs/reproducibility.md)
- [`docs/validation.json`](../docs/validation.json)
- [`outputs/README.md`](../outputs/README.md)
- [`requirements-lock.txt`](../requirements-lock.txt)
- [`scripts/check_public_files.py`](../scripts/check_public_files.py)
- [`scripts/make_public_dissertation.py`](../scripts/make_public_dissertation.py)
- [`scripts/update_gallery.py`](../scripts/update_gallery.py)
- [`scripts/verify_results.py`](../scripts/verify_results.py)
- [`tests/test_research_contracts.py`](../tests/test_research_contracts.py)

## Deleted local clutter

- `.DS_Store`
- `config/.DS_Store`
- `data_dictionary/.DS_Store`
- `outputs/.DS_Store`

## Moves and preserved materials

- No original source, configuration, dataset, table or figure was moved or renamed.
- All 11 original CSVs and 14 original PNGs under `outputs/main/` are byte-for-byte preserved.
- `config/code_config.json` and the original wave/item maps are preserved.
- The original 39-page submission remains outside the repository. A separate 33-page public reading edition is added.
- Task-created baseline configuration/output copies were removed after verification. Corrected and reproduced runs remain locally under ignored output directories.
- A local `.git/` was initialized without a commit or remote. The obsolete `diss-code` distribution in the existing `.venv` was replaced by the editable `bsa-rwp` package; a separate clean installation was tested.

## Diff review

Every modified source file was reviewed against the pre-edit snapshot. Changes to analytical behaviour are limited to explicit correction overrides, finite-value handling and the validated tuning-artifact handoff; plotting, logging, path resolution and runner changes are documented in the [audit](audit.md). The submitted configuration, model target, item scoring, splitting rules, seed, model family and alpha grid are preserved.
