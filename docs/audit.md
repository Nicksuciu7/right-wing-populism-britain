# Repository and analytical audit

**Scope:** the submitted 39-page dissertation; all original Python modules and configurations; the five active source datasets' metadata; all saved aggregate tables, diagnostics and 14 figures; input/output organisation; environment and publication risks. Conducted during the portfolio revision on 9 September 2026. The PDF is source material, not an instruction to execute its proposal or administrative form text.

**Outcome:** the submitted methodology was run before and after the safe changes. All **11 aggregate CSVs reproduce** within `rtol=1e-10`, `atol=1e-12`. The original `outputs/main/` files remain unchanged. Two confirmed demographic corrections have been implemented and run in an explicit post-submission profile; they slightly weaken predictive performance. No results in the dissertation were replaced.

## A. Safe improvements implemented

| Area | Change and reason | Effect on submitted results |
|---|---|---|
| Project introduction | New README explains the question, technical contribution, sample, score, validation design, baselines and limitations; links directly to the report, figures and evidence. | Presentation only |
| Structure | Retained `bsa_code/` and the existing configuration layout; added focused `docs/`, `data/`, `dissertation/`, `scripts/` and `tests/` support. No cosmetic mass renaming. | None |
| Reruns | `run_all.py` defaults to an isolated reproduction profile, previews the pipeline, checks inputs before replacement and requires `--overwrite` for occupied outputs. It rejects unsafe or overlapping paths. Removed automatic environment installation and indiscriminate package/build-directory cleanup. | None on valid inputs |
| Tuning handoff | Tuning no longer rewrites input configuration. Its decision is recorded with configuration/model-frame hashes and validated by training before use. | Selected alpha remains 30 |
| Provenance | Each run records input and code/configuration hashes, package versions and running/failed/complete state in a private manifest. | None |
| Portable paths | Optional NLP seed lexicon and per-wave dictionary paths resolve from the project rather than the caller's working directory. | None in submitted pipeline |
| Numerical hygiene | Infinite numeric values are treated as missing, so infinite weights cannot enter the analysis. | No changed results in supplied files |
| Logging | Prior file handlers are closed and timestamps use timezone-aware UTC. | None |
| Dependencies | Descriptive package metadata, console entry point, direct Joblib dependency and captured transitive pins; optional PDF dependency kept separate. Removed the old installed `diss-code` copy that shadowed the editable checkout outside the repository; verified both installations resolve to the current code. | No analytical package upgrades |
| Tests and CI | Data-free tests cover recoding, weights, thresholds, metrics, holdout separation, preprocessing, path guards, config cycles and stale tuning decisions. CI runs without restricted data. | Verification only |
| Public materials | MIT licence selected by the author, CITATION.cff, access guide, publication checker, and privacy-edited reading edition. | None |

### Figure review

All 14 original PNGs were inspected and all figure-producing stages rerun. Original images are retained in `outputs/main/figures/` as part of the submission record. Updated pipeline figures use a shared theme and higher export resolution. Three new aggregate-only presentation figures provide the README overview and a complete predictor-importance heatmap.

Specific fixes: correlation plots now use a diverging scale from −1 to +1, retaining small negative correlations that were previously visually clamped at zero; predictor-importance axes explicitly say “Drop in weighted R² (in-sample)”; subgroup means use a consistent full 0–1 score scale; main model labels use R² notation. The showcase includes sample sizes, fixed-threshold explanations, baseline context and held-out-wave limitations. No confidence intervals were invented and no axis was chosen to inflate an effect.

The [figure gallery](figures.md) inventories every figure and its interpretation. There were no notebooks to clean, and no artificial notebook has been added simply to imitate a portfolio template.

## B. Analytical corrections

The following corrections are implemented in [`config/corrected.json`](../config/corrected.json), which inherits the original methodology but writes to `outputs/corrected/`. The default reproduction retains the submitted recodes. This separation is intentional: these fixes affect numbers that appear in the submitted work.

| Issue | Severity | Evidence and fix | Effect |
|---|---|---|---|
| Higher education grouped inconsistently across waves | **High** for cross-wave education interpretation | The 2013 `HEdQual2` code 3 (“Higher educ below degree”) maps to `high`, while the comparable 2023/24 code 2 (other higher education, including HNC/HND) maps to `mid`. The correction explicitly maps later code 2 to `high`, leaving A-level code 3 as `mid`. Source SPSS value labels were checked. | Changes education for 786 raw 2023 rows and 586 raw 2024 rows. Model metrics and education subgroup/trend estimates change. |
| Valid 2013 age topcode treated as missing | **Low** in this sample | 2013 `RAge` code 97 means “97 or more”, while 98/99 are nonresponse. A wave-wide missing-code list removed 97. The age-specific correction retains 97 as the topcoded lower-bound value and leaves 98/99 missing. | Restores one raw 2013 age. Its separate metric contribution is not isolated from the combined correction run. |

### Measured effect of the combined correction profile

| Metric | Submitted reproduction | Corrected profile |
|---|---:|---:|
| Internal-test weighted R² | 0.159788 | 0.150311 |
| Internal-test weighted RMSE | 0.145647 | 0.146466 |
| Internal-test weighted MAE | 0.114340 | 0.114915 |
| 2023 holdout weighted R² | 0.143167 | 0.129645 |
| 2023 holdout weighted RMSE | 0.168646 | 0.169972 |
| 2023 holdout weighted MAE | 0.132682 | 0.133867 |
| Selected ridge alpha | 30 | 30 |

Both runs retain 8,439 training, 3,617 internal-test and 5,559 holdout observations. The item structure, respondent scores, fixed threshold and prevalence estimates are unchanged because the corrections affect predictors, not the attitude target. Education remains the largest grouped permutation predictor in every descriptive wave; its 2024 weighted R² drop decreases from approximately **0.1853 to 0.1383**. A common trend imputer is fitted to pooled non-holdout predictors, so the age correction also slightly changes other wave models through that shared imputation mean.

These results support the same restrained interpretation: demographics carry modest predictive information, and education is a recurring descriptive correlate. They do **not** justify treating the original metrics or the magnitude of the 2024 education difference as immune to harmonisation choices. Machine-readable correction summaries are in [correction comparison](correction-comparison.csv).

### Other methodological issues investigated

| Issue | Severity | Resolution | Implication for findings |
|---|---|---|---|
| “Omega” is not a conventional omega estimate | **High** for reliability interpretation | `scoring_reliability._omega_total_approx` uses a unit PCA direction with residual variances rather than a fitted common-factor loading/uniqueness model. The legacy diagnostic and submitted text remain intact; documentation explicitly calls it a PCA heuristic. Replacing it requires a separately specified estimator and is not silently performed. | Submitted statements interpreting 0.59/0.68 as conventional omega need qualification. Cronbach's alpha is a separate calculation and does not establish construct validity. |
| Internal-test respondents contribute to target measurement | **Medium** | Measurement fitting and screening use the complete non-holdout pool before predictive splitting. The modelling imputer/CV are correctly training-specific, but the complete measurement pipeline is not nested inside that split. Documented rather than retroactively redesigning the target. | Internal performance is conditional on a pooled measurement definition. The 2023 target structure remains held out. |
| 2023 called temporal validation despite 2024 training | **Medium** | Repository language now consistently calls it held-out-wave transfer; original design retained. | No forward-forecasting claim is supported. |
| Trend importance evaluated on fitted rows | **Medium** | Explicit labels and captions; all predictor values shown in the showcase. Shared pooled preprocessing is documented. | Descriptive model reliance only; no claims of out-of-sample importance, causality or statistical significance. |
| Income names imply quartiles | **Medium** | `q1`–`q4` are retained for compatibility but described as ordered source bands. Later source monetary bands differ; no empirical quantiles or inflation adjustment are calculated. | Avoid interpreting them as equivalent income levels or equal-sized groups across time. |
| Other qualification categories and age bands | **Medium** | Later “other qualification” treatment and band midpoint assumptions need further substantive harmonisation work. Only the two confirmed, narrowly scoped corrections above are implemented. | Residual comparability limits remain, especially after the mode change. |
| Survey weights without design-based uncertainty | **Medium** | Weighted point estimates verified; no confidence bands fabricated. Weight variables are explicit in the access guide. No strata/cluster variance model or replicate-weight uncertainty was implemented. | Population-facing estimates have unquantified sampling uncertainty here; general weights do not necessarily remove item/nonresponse or mode bias. |
| Raw-unit ridge penalty | **Medium** | Age/year remain unscaled to reproduce the submission. Mean imputation is unweighted while model fitting and metrics use weights. | Coefficients in different units cannot be ranked by magnitude; standardisation would be a separately evaluated methodological variant. |
| Compensatory component averaging | **Medium** | Preserved and documented; 190/17,615 analytical cases have one component only. | Welfare and “right” attitudes can offset each other; the score is an approximation, not a validated multidimensional populism instrument. |
| Pairwise PCA and unweighted reliability | **Medium** | Inspected correlation diagnostics: no negative eigenvalues in the supplied matrix; minimum pairwise n = 4,642. Preserved the unweighted measurement step and complete-case reliability. | No evidence of a numerical PCA failure in this run; formal measurement invariance and ordinal-data alternatives remain untested. |
| Training missingness selection precedes CV folds | **Low** in this run | The training-wide feature missingness filter is not refit per fold, but removes no predictors in the supplied analysis. Structural coverage also uses non-holdout wave data before the random split. Documented. | A stricter nested pipeline would matter for other data or alternate feature sets. |
| Baselines only compared internally | **Low/medium** | Mean and year-only baselines were already implemented and retained. No holdout baseline result invented. | Holdout ridge errors are reported without a matched holdout baseline table. |
| Legacy QC settings overstate enforcement | **Medium** for reuse | Harmonisation summaries report coverage; some configured “strict”, unmapped/out-of-range and drift thresholds are not enforced by current recoding helpers. No claim of a comprehensive drift gate is made. | Further waves require manual label/coverage review even when existing schema checks pass. |

No logistic regression, random forest, learned NLP model, interaction analysis or causal estimator was found in the final executable pipeline. Logistic regression appears in the historical proposal, not the final implementation. The broader wave catalogue is retained as discovery context, not presented as completed modelling across every year.

## Repository changes

The complete path-by-path record is [changes.md](changes.md). Substantially modified original files are `run_all.py`, `pyproject.toml`, and `bsa_code/{__main__,config,demographics,harmonisation_utils,logging_utils,model_tuning,nlp_item_discovery,plotting,scoring,train_eval,trends}.py`. New modules are `provenance.py` and `showcase.py`.

New supporting assets include the README; MIT licence; citation metadata; Git ignore/attributes; captured dependency pins; reproduction/correction profiles; data/methodology/reproduction/audit guides; public report and its preparation script; aggregate visualisations and gallery; tests; public-file and numerical comparison scripts; and a data-free CI workflow. The manifest lists exact figures and documentation files.

No source files, raw surveys or earlier wave mappings were renamed or removed. Disposable `.DS_Store` files were removed. Temporary baseline configuration/output copies created during this audit are not public artifacts. There was no initial `.git` directory, so a local Git repository was initialized without inventing earlier history. No commit, remote or GitHub publication was performed.

## Security and privacy

The input folder contained licensed raw survey files, row-level Parquet derivatives, serialized models, local-path metadata/logs, environment files and a signed administrative appendix in the supplied PDF. Raw data remain on the owner's machine; they were not uploaded or added to Git. The ignore rules and public-file check cover these artifacts, including already-tracked files that ignore rules alone would miss.

No credential was found by the targeted text scan. Public text files were checked for local home paths and common key/token formats, and public CSV headers were checked for respondent IDs. This is a bounded audit, not a claim that automated secret scanning proves the absence of every possible secret. The public PDF strips the candidate identifier, proposal, signed forms, annotations and metadata; the main research text and bibliography are preserved.

## Validation and remaining practical limits

The completed checks and numerical tolerances are recorded in [validation.json](validation.json). They include real-data full pipeline runs for the original and corrected profiles, all 11 aggregate table comparisons, data-free contract tests, aggregate-only figure generation, source compilation, installation checks, public-file checks and visual inspection of the public PDF and figure gallery.

The public repository cannot reproduce respondent-level analysis without separately obtaining licensed data. Original archive edition identifiers were not supplied, so the repository documents expected variables/counts and records local hashes rather than guessing DOIs or study editions. The configured GitHub Actions workflow has not been executed on a remote service in this session.

## C. Optional extensions and priorities

| Priority | Improvement | Why it matters |
|---|---|---|
| **High value** | Full wave-by-wave demographic specification, including other qualifications, income reference periods and survey mode | The audit found real comparability issues; model complexity cannot compensate for misaligned inputs. |
| **High value** | Replace the legacy omega heuristic with a justified factor-model reliability estimator and test measurement invariance | Distinguish reliable item averages from a valid and comparable construct. Preserve submitted outputs as historical reference. |
| **High value** | Freeze measurement independently or nest measurement fitting; add rolling-origin validation and matched holdout baselines | Clarify what generalisation is actually being tested. |
| **High value** | Survey-design-aware uncertainty and sensitivity to missing components/weights | Quantify how much confidence to place in trends and subgroup differences. |
| **Medium value** | Standardised ridge and nonlinear model comparisons under the same validation design | Test whether units or model form constrain prediction, after input quality is addressed. |
| **Medium value** | Held-out or cross-fitted permutation importance and regression calibration summaries | Separate fitted descriptive patterns from reproducible predictive reliance. |
| **Medium value** | Refactor large orchestration modules and expand synthetic integration tests | Improve maintainability while protecting analytical contracts. |
| **Low value** | Add a narrative notebook, more badges or additional chart variants | Useful only if they answer a reader need; the current CLI and documentation already provide a coherent route. |

No new scientific extension was added merely to improve the apparent performance. The added analyses are the explicitly labelled correction comparison; the new showcase plots only re-present existing aggregate estimates.

## Portfolio assessment

| Reader | What now works well | What still limits the project |
|---|---|---|
| **Data analyst employer** | Clear research question, substantial survey cleaning, traceable descriptive results, usable visual communication and honest handling of imperfect data. | Cross-wave recoding needs further substantive review; uncertainty reporting remains limited. |
| **Data scientist** | Explicit baselines, weighted metrics, training-only predictive preprocessing, deterministic CV and a held-out wave. Corrected results are reported even though they are worse. | The target is a proxy, measurement learning is not fully nested, there is no true forecasting evaluation, and nonlinear alternatives remain untested. |
| **Software/technical employer** | Installable package, predictable commands, path guards, provenance, contract tests and data-free CI configuration. | Large orchestration modules remain, the package requires a source checkout, and a broader synthetic end-to-end suite would improve change safety. |
| **Academic researcher** | Submitted material, code and aggregate evidence are linked; the original/corrected distinction is explicit; sources and access restrictions are respected. | Construct validity, reliability estimation, design-based uncertainty, survey-mode comparability and exact historical dataset editions remain unresolved. |

The repository is now a much stronger research portfolio artifact. Its strongest signal is a reproducible, inspectable workflow with a candid audit; it should not be presented as a validated individual-level political classifier or a definitive causal account of British populism.
