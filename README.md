# Power outages reveal inequalities in electricity service: from global to local scales

Code and compact processed data accompanying the manuscript revised on 27 October 2026, with supplementary tables from 25 October 2026.

The default workflow recomputes the core historical, recent, regional and Brazilian poverty comparisons and redraws all four main figures. It uses the manuscript's actual analytical populations: 152 historical paired benchmarks, 31 countries in the recent comparison, 703 Brazilian service areas, 40 cause profiles (36 paired), and 23 countries in the supported household comparison.

## Run

Use Python 3.12 in a new environment:

```bash
git clone https://github.com/FX794726198/global-electricity-service-inequalities.git
cd global-electricity-service-inequalities
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_package.py
python reproduce.py
python verify_results.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The workflow reads packaged inputs and writes to `outputs/reproduction/`; no data download or account is needed. Figures are exported as PNG, PDF and SVG in `outputs/reproduction/figures/`. Per-step logs and a machine-readable execution report are retained beside them. The scripts use fixed seeds; fonts and rendering libraries can affect image bytes without changing the estimates. Arial was used for the manuscript; a system substitute may be used when it is unavailable.

To also redraw the supported supplementary figures:

```bash
python reproduce.py --supplement --output outputs/with_supplement
python verify_results.py --output outputs/with_supplement
```

Use `--resume --steps STEP ...` to continue an existing output directory, or `--help` to list individual steps. A fresh run never overwrites the packaged inputs. The supplementary plotting workflow redraws all 18 supplementary figures from the supplied processed observations and result tables.

## Contents

| Location | Contents |
| --- | --- |
| `reproduce.py` | Isolated, logged execution of the supported workflow |
| `verify_package.py`, `verify_results.py` | File-integrity checks and comparisons with frozen analytical results |
| `study/10_manuscript/` | Analysis inputs, frozen result tables and the exact version-specific analysis/plotting programs used by the workflow |
| `study/07_analysis/` | Source-processing programs and additional regional/household aggregate tables |
| `study/08_database_expansion/` | Source-resolved cause-profile inputs used in the final integration |
| `data/supplementary_tables/` | Tables S1–S29 and the unnumbered evidence map, transcribed from the current supplement; `index.json` retains their captions |
| `metadata/provenance.csv` | Original project-relative source paths, source hashes, packaged hashes and portability changes |
| `metadata/SHA256SUMS` | Checksums of the distributed files |
| `metadata/validation.json` | Results of the release verification |
| `docs/DATA.md`, `docs/REPRODUCIBILITY.md` | Data sources, units, coverage and limits of reproduction |

Historical directory/version names are retained because the final manuscript combines analyses developed at different stages. The entry point and figure mapping, rather than directory dates, identify the current results. Superseded prediction experiments and large raw archives are excluded.

## Data and interpretation

The package starts from processed, source-specific observations and aggregates. It includes the inputs needed by the supported numerical workflow, rather than the full collection of downloaded reports and individual survey records. Household respondent-level microdata must be obtained from Afrobarometer and World Bank Multi-Tier Framework providers under their access conditions. Official URLs, periods and reporting populations are listed in [Table S1](data/supplementary_tables/Table_S01.csv); additional cause-profile provenance is retained in the figure inputs.

Source-native units are preserved. In particular, record-hours in the 2024–2025 Brazilian fine-cause archive are not customer-hours; transmission energy-not-supplied shares are not outage durations; historical business-city benchmarks are not nationally representative household measurements. Missing or unreported cause categories are not silently converted to observed zeros. See [the data guide](docs/DATA.md).

The supplemental CSV tables are frozen manuscript tables, not evidence that every table has been re-estimated from raw data by the entry point. Likewise, plotting a supplied household estimate does not rerun the provider-level survey analysis. The distinction is documented in [the reproduction guide](docs/REPRODUCIBILITY.md).

Third-party data retain their providers' access and reuse conditions. This repository does not grant a blanket license to the underlying third-party material. When using these materials, cite the manuscript, this repository and the relevant data providers. No archival DOI is claimed.
