# Methodology and implementation

The authority for what was submitted is the [dissertation](../dissertation/dissertation-public.pdf). This guide traces the implementation and records its practical boundaries. Audit corrections are separately configured and evaluated.

## Design and sample

Five repeated cross-sectional BSA waves are pooled: 2011, 2013, 2019, 2023 and 2024. The supplied source files contain 19,477 rows. The intermediate joined frame has 19,244 rows, while the valid-score/weight modelling sample has 17,615. The latter comprises 8,439 training rows, 3,617 internal-test rows and 5,559 held-out 2023 respondents.

Year is a numeric predictor in the pooled ridge model. The 2023 wave is held out despite 2024 appearing in training, so this design tests transfer across waves rather than prospective temporal forecasting. It cannot identify the effect of the financial crisis, Brexit or other historical events.

## Item discovery and harmonisation

`inventory.py` and `meta_utils.py` inspect SPSS labels and column availability. Optional `nlp_item_discovery.py` searches RTF dictionary text using the seed lexicon; its output is a candidate list, not an automatically validated construct. `attitude_map.py` combines the base map and year-specific overrides.

`ingest.py` reads the necessary columns. `attitudes.py` and `harmonisation_utils.py` mask configured missing/excluded/out-of-range responses, scale valid responses to 0–1 and apply explicit reversal. `demographics.py` maps the changing source codes to common categories. Diagnostics report non-missing coverage. Some legacy QC configuration keys suggest stronger thresholds than the current diagnostics actually enforce; they must not be interpreted as a complete harmonisation validation system.

Income `q1`–`q4` labels are ordered source bands, not freshly calculated quantiles. Age uses numeric values including documented band substitutions for the later waves, not exact ages for every respondent. Religion-related missing attendance can be recoded to never-attending for respondents reporting no religion. These are substantive harmonisation choices.

## Measurement construction

The candidate construct combines political distrust, authoritarianism, immigration and welfare attitudes. Reference years are 2011, 2013, 2019 and 2024. Shared availability is required across those years, followed by a within-block maximum-absolute-correlation screen at 0.1.

The retained 17-item pairwise Pearson correlation matrix is unweighted. PCA eigen-decomposition is applied to that matrix, followed by a scree-elbow rule and varimax rotation. Assignment requires an absolute loading of at least 0.3 and a gap of at least 0.1 over the next strongest loading. Two components are retained, with 10 and 7 items. Component signs are aligned for interpretation; scoring uses the already-oriented item responses, not the signed PCA projection.

Pairwise deletion allows respondents with incomplete item batteries to contribute. The original reference matrix has minimum pairwise n = 4,642 and minimum eigenvalue approximately 0.28184, so no negative eigenvalues occur in this run. Pairwise correlation matrices need not be positive semidefinite for other data; the numerical handling is not a substitute for reviewing missingness and measurement stability.

A respondent needs `ceil(0.6 × item_count)` observed answers for each component: six of ten “right” items and five of seven welfare items. Component scores are observed-item means. The final `rwp_score` is the unweighted mean of available component scores. No item imputation or loading-weighted factor scoring is used. Exactly 190 of the 17,615 analytical observations have only one valid component.

The fixed weighted 80th-percentile cutoff is approximately 0.6743435953, estimated on non-holdout waves. `RWP_top_20` uses an inclusive comparison (`>=`), so ties and weighted interpolation mean neither each wave nor the pooled set must contain exactly 20% positive labels. It is a descriptive threshold label, not the target of a classifier.

Complete-case Cronbach's alpha is recorded for pooled and per-wave components. The saved `omega_total_approx` uses a unit PCA direction and residual variance; it is not a standard factor-model omega estimator. Its values should not be interpreted as conventional omega or evidence of construct validity. See the [audit](audit.md).

## Predictive modelling

`model_frame.py` joins demographics and scores using respondent ID and wave year and enforces one-to-one keys. Structural coverage uses the non-holdout waves. All 13 main predictors pass; attitude items and derived outcome labels are not model inputs.

After reserving 2023, the other waves receive a random 70/30 split with seed 42. The primary split is **not year-stratified**. A second missingness filter is learned on the training subset (70% cutoff). Numeric means are learned within each fitted preprocessing workflow; categorical missingness is encoded as a level. One-hot encoding drops the first configured category and ignores unknown categories. Numeric age and year remain unscaled, which affects ridge regularisation and means coefficient magnitudes are not directly comparable across units.

Five-fold CV is year-stratified within the training subset, with a fallback to K-fold for insufficient year counts. Each fold refits preprocessing. The fixed alpha grid is 1, 3, 10, 30, 100, 300, 1000 and 3000. Selection minimises mean weighted RMSE, breaking ties by weighted MAE and then higher alpha. Both submitted and corrected profiles select 30. Internal-test results are recorded after CV selection; they do not select alpha. The holdout is untouched by tuning.

Three models share the training/test partition: a weighted training-mean baseline, weighted year-only linear regression and weighted ridge regression. Main reported metrics are weighted RMSE, MAE and R²; unweighted metrics and internal-test Pearson/Spearman correlations are also retained. Predictions are not clipped to the 0–1 target range. The saved comparison evaluates baselines internally; it does not provide a holdout baseline table.

The original numeric target construction uses all non-holdout respondents, including the random-test subset, before predictive splitting. Thus the internal test measures prediction conditional on a pooled measurement definition; it is not validation of the entire measurement-learning pipeline from scratch. A nested or externally fixed measurement design would answer a stricter question and is left as a separate research extension.

## Descriptive trends

The 2023 wave is excluded. Trend predictors must be usable in at least three waves under the stricter missingness rule, and year is removed. A common preprocessor is fitted on all non-holdout trend rows, including a pooled numeric imputation mean. Separate weighted OLS models are then fitted within each wave.

Grouped permutation importance shuffles each original predictor before transformation, so a categorical predictor's encoded columns are assessed together. Each predictor receives 20 permutations with seed `42 + wave_year`. The change in weighted R² is evaluated on the same rows used to fit that wave's model. It is **in-sample descriptive reliance**, sensitive to correlated predictors and the shared preprocessing definition; it is not a causal effect, a forecasting metric or a significance test. Correcting a pooled imputation input can slightly affect other waves' summaries.

Subgroup plots show weighted means, with unweighted group counts in their CSVs. Survey weights are used in regression fitting, metrics, thresholds, prevalence and subgroup means, but not the PCA correlations or complete-case reliability calculation. Survey weights alone do not supply design-based standard errors. No confidence intervals, calibration analysis, formal invariance tests, random forest results or interaction models are part of the submitted executable pipeline.

## Module map

| Responsibility | Modules |
|---|---|
| Entry points and configuration | `run_all.py`, `__main__.py`, `config.py`, `validate_config.py` |
| Source metadata and discovery | `inventory.py`, `meta_utils.py`, `rtf_parser.py`, `nlp_item_discovery.py` |
| Harmonisation | `ingest.py`, `attitude_map.py`, `attitudes.py`, `demographics.py`, `harmonisation_utils.py`, `recoding.py` |
| Measurement and scoring | `scoring.py`, `scoring_pca.py`, `scoring_scores.py`, `scoring_thresholds.py`, `scoring_reliability.py` |
| Prediction and descriptive analysis | `model_frame.py`, `model_preprocessing.py`, `model_tuning.py`, `train_eval.py`, `trends.py` |
| Outputs and reproducibility | `plotting.py`, `output_layout.py`, `logging_utils.py`, `utils.py`, `provenance.py`, `showcase.py` |

The largest orchestration modules remain intact to avoid changing the submitted computational logic during presentation work. Further decomposition should be guided by the current numerical regression checks, rather than stylistic renaming alone.
