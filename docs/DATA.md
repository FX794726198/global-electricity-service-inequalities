# Data guide

## Source register

[Table S1](../data/supplementary_tables/Table_S01.csv) is the source register from the current supplement. Each row gives a provider URL, reporting population, period and measurement scope. The cause-profile files additionally retain source locators, native units, crosswalks and reporting flags. URLs are the study's recorded sources; the reproduction workflow uses local snapshots and does not depend on their continued availability.

The package contains processed public-source observations, administrative aggregates, map geometry and frozen model estimates. It excludes household respondent records, survey `.sav`/`.dta` files, full downloaded report archives, private correspondence, credentials and manuscript documents.

## Main analytical inputs

Paths below are relative to `study/10_manuscript/`.

| Analysis | Input location | Unit / scope |
| --- | --- | --- |
| Historical demand and reliability | `20261003_global_blackout_inequalities_v33/core_reproducibility/inputs/`; `20261003_global_blackout_inequalities_v33/source_data/gradient_v31_benchmark_inputs.csv` | Economy-year demand and historical business-city/utility benchmarks; 152 positive paired frequency/duration observations |
| Recent validation | `20261003_global_blackout_inequalities_v33/core_reproducibility/upbeat_extension/inputs/utility_year_all.parquet` | Source-year observations retained with scope and admission flags; 31 countries across 2021–2023, not a balanced 31-country panel |
| Regional partitions | `20261003_global_blackout_inequalities_v33/core_reproducibility/class_gap_extension/inputs/` and `data/frozen_accounting_inputs.csv` | Matched administrative populations; all categories remain together within a resampled unit |
| Türkiye | `20261003_global_blackout_inequalities_v33/core_reproducibility/turkiye_extension/data/` | Published settlement-group indices and native cause totals; no invented weather classification for unresolved labels |
| Brazilian poverty comparison | `20261003_global_blackout_inequalities_v37/analysis/poverty_priority/inputs/` | 703 service areas, 275 municipalities, 24 states, 24 months in 2024–2025, joined to Census 2022 resident-income context |
| Cause-profile atlas | `20261009_global_blackout_inequalities_v40/source_data/fig2_profiles.csv`, `fig2_shares.csv`, `fig2_sources.csv` | 40 reporting profiles; 36 paired occurrence/consequence compositions and four occurrence-only profiles |
| Household service partition | `20261003_global_blackout_inequalities_v33/source_data/Fig03_same_population_service_components.csv` | Country-level estimates for 23 supported countries; original individual interviews are not supplied |

`study/07_analysis/` provides additional aggregate data and the original processing/model programs for the definition audit, source-native causes, poverty contrasts, household surveys and connection barriers. Those upstream programs can refer to provider downloads outside this compact package. They are not all executed by `reproduce.py`.

## Variables and missingness

- `F` / `frequency`: interruptions per customer-year when the provider reports customer-based indices. Other source-native occurrence measures are explicitly labelled in the atlas.
- `H` / `annual_hours`: interruption hours per customer-year for the same population and category as `F`.
- `D` / `duration`: `H / F` within a supported positive-frequency population, in hours per interruption. An undefined denominator remains undefined.
- `gap_h`, `gap_H` and component fields: an annual interruption-hour difference and its frequency/duration accounting components. The direction is specified by each analysis: lower minus upper income, higher minus lower poverty, or rural minus urban.
- `low_income_share_pct`: municipal share of residents in the specified low-income Census 2022 category. This is area context and does not assign a specific outage to an interviewed household.
- `occurrence_share_pct`, `burden_share_pct`: percentages within a profile's own published denominator. The `occurrence_unit`, `burden_unit`, `paired`, `category_reported` and source-scope fields must accompany them.
- Household proportions are on the 0–1 scale unless a column explicitly uses percentages or percentage points (`pp`). Country-equal summaries do not constitute a pooled global household estimate.
- Empty CSV cells / Parquet nulls retain missing or inapplicable values. In the atlas, absence of a reported category is distinct from a reported zero; occurrence-only profiles have no burden distribution.

The Brazilian 703-area fine-cause table represents 3,383,191 published records. The lower cause reconstruction contains 3,383,115 records, leaving 76 zero-duration records unallocated; cause-specific upper bounds overlap and must not be added. The corresponding record-duration total is 23,173,145.8144 record-hours. This is separate from the customer-based regulatory indices used for the poverty-gap decomposition.

## Provider-level reproduction

Obtain the original files from the providers in Table S1 and preserve their native variable definitions, weights and access conditions. The source-processing scripts document expected file paths, filters, joins and outputs. Optional Python dependencies are in `requirements-source-processing.txt`. The compact package does not include every prerequisite of those acquisition workflows and does not claim a fresh download-to-result audit of all original providers.

The official provider terms continue to apply. Public accessibility of a URL does not imply that all linked microdata or source documents can be redistributed without conditions.
