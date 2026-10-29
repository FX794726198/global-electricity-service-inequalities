# Reproduction coverage

## Executed numerical workflow

| Step | What is recomputed or checked |
| --- | --- |
| `core_checks` | 346 numerical/source-accounting checks, historical decomposition, definition subsets, household country partition and rank correlations |
| `historical` | Fixed-group historical means, paired bootstrap intervals and frequency/duration decomposition |
| `regional`, `regional_checks` | Matched regional class partitions, bootstrap results and independent reconstruction from the frozen inputs |
| `turkiye_checks` | Settlement and native-category accounting identities against published totals |
| `recent` | Source admission, annual country comparisons, fixed-cohort sensitivity, paired bootstrap intervals and economic gradients |
| `priorities` | Same-population average-burden versus gap-contribution rankings and weighting sensitivity |
| `brazil_poverty` | 703-area monthly reconstruction, poverty grouping, municipality bootstrap, category contrasts and missing-category checks |
| `cause_profiles` | Final cause-profile integration from source-resolved profile/share inputs; 40 profiles, 36 paired |
| `figure1`–`figure4` | Main-figure rendering; Figure 4 also recomputes the country-level household partition bootstrap |

`verify_results.py` compares regenerated tables with frozen author result tables, with stated floating-point tolerances. It also checks run statuses and figure exports. This is a package-reproduction check; it does not independently validate every original source or establish the manuscript's scientific conclusions.

## Supplement

`--supplement` redraws all 18 supplementary figures. Most of these use supplied aggregate/coefficient tables. S6 follows the recomputed recent analysis; S8 follows the recomputed matched-priority analysis. Europe response curves in S7 are refitted to the supplied 11-system observations. Provider-level survey estimation and all original report extraction are outside this command.

All 29 numbered supplementary tables and the unnumbered evidence map were transcribed from the supplied supplementary document. Their captions are in `data/supplementary_tables/index.json`; they have not all been newly estimated by this release workflow.

## Changes made for distribution

The original statistical algorithms, seeds, group definitions and scientific input values are retained. Packaging changes replace machine-specific root paths and package search paths, reuse a single world-geometry file and compact snapshots of the original plotting-only boundaries, exclude macOS resource-fork files from inventories, and expose an isolated runner. Figure 1's current axis labels and rank-correlation annotations are applied in its data-to-figure program. Original file hashes and packaged hashes are recorded in `metadata/provenance.csv`.

The package reproduces analytical values and figure content. It does not promise byte-identical PDFs/PNGs across font, graphics-library or operating-system versions. It does not edit the manuscript or regenerate its Word layout.

## Verification environment

The release was exercised with Python 3.12.14 on macOS arm64. Package versions are pinned in `requirements.txt`; execution statuses, comparison results and platform details are recorded in `metadata/validation.json`. A different platform should first run `verify_package.py`, then the default workflow and `verify_results.py`.

Expected checks include 152 historical benchmarks; historical lowest-minus-highest annual-hour gap 133.8329947368 h, partitioned into 96.1595159937 h frequency and 37.6734787431 h duration components; 31 countries across the recent panel; 703 Brazilian areas; and the 23-country household gap of 22.4903834925 percentage points. The Brazilian primary area-equal gap is 19.05 h, including 14.61 h in the ordinary-unplanned category (rounding may differ at the final digits).
