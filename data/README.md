# Survey data: access and local layout

The analysis uses NatCen's **British Social Attitudes (BSA)** survey, obtained through the UK Data Service. These are repeated cross-sectional surveys: respondents are not followed longitudinally. The study concerns Britain, rather than a UK-wide panel.

## Obtain the five required waves

1. Start with the [BSA series catalogue](https://datacatalogue.ukdataservice.ac.uk/series/series/200006?id=200006) or [NatCen's BSA overview](https://natcen.ac.uk/british-social-attitudes).
2. Find the full survey datasets for **2011, 2013, 2019, 2023 and 2024**. Register and obtain access for your own project under each collection's current conditions. Check the [UK Data Service access policy](https://ukdataservice.ac.uk/help/access-policy/types-of-data-access/) and [safeguarded-access guidance](https://ukdataservice.ac.uk/find-data/access-conditions/safeguarded-access/).
3. Download the SPSS versions and their documentation. Teaching subsets are not substitutes for the full files: the required variables may be absent.
4. Extract the survey files locally and rename or copy them to the locations below. These are this project's local names, not guaranteed archive download names. Keep the original catalogue citation, study number, edition and download date in your own access records.
5. Run `python -m bsa_code --config config/reproduce.json validate_config` before analysis.

| Wave | Expected file | ID variable | Weight variable | Raw rows in supplied files |
|---|---|---|---|---:|
| 2011 | `raw_sav/bsa11.sav` | `Serial` | `WtFactor` | 3,311 |
| 2013 | `raw_sav/bsa13.sav` | `Serial` | `WtFactor` | 3,244 |
| 2019 | `raw_sav/bsa19.sav` | `Sserial` | `WtFactor` | 3,224 |
| 2023 | `raw_sav/bsa23.sav` | `Serial_scrambled` | `BSA23_final_wt` | 5,578 |
| 2024 | `raw_sav/bsa24.sav` | `serial_EUL` | `BSA24_final_wt_GB18` | 4,120 |

The supplied extracts total **19,477 records**. The saved model frame retains 19,244 rows, including rows without a score; the final valid-outcome modelling sample is **17,615**. Do not confuse the intermediate frame size with the analytical sample size. File editions can differ: the run manifest records the exact local input hashes used, but historical archive edition identifiers were not supplied with this repository and have not been guessed.

SPSS labels drive metadata checks and inventory. The core five-wave pipeline does not require the RTF dictionaries. To rerun the optional dictionary-based candidate discovery, also place `bsa11_data_dictionary.rtf`, `bsa13_data_dictionary.rtf`, `bsa19_data_dictionary.rtf`, `bsa23_data_dictionary.rtf` and `bsa24_data_dictionary.rtf` in `data_dictionary/`.

## What is kept out of Git

Raw survey files, RTF dictionaries, respondent-level Parquet files, model frames, scores, model objects and logs remain local. `.gitignore` excludes the input directories and all build/rerun outputs. The public repository contains reviewed aggregate tables and figures only. Derived respondent records remain subject to the underlying data conditions; removing names is not permission to redistribute them.

The original workspace also contains 2007–2024 source waves and mappings. The final dissertation uses only the five configured waves above. `config/waves/` is a retained discovery-era catalogue and is **not loaded** by `config/code_config.json`. Do not interpret those files as evidence of an 18-wave analysis.

Income values `q1`–`q4` are historical code names for four ordered source bands. The program does not calculate income quartiles, adjust monetary bands for inflation, or establish equal-sized groups. Survey age bands and changed survey modes also limit comparability. See [methodology](../docs/methodology.md) and [audit](../docs/audit.md).
