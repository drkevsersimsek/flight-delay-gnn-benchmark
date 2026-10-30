# Flight Delay GNN vs MLP Benchmark

Code and trained model checkpoints for the paper *"When Does Graph
Structure Help in Airport Delay Forecasting? A Controlled GNN–MLP
Benchmark"*. This repository contains the full data processing and
model training pipeline; large processed data files are hosted
separately on Zenodo (see [Data](#data) below).

**Archived release:** this repository is archived at
[https://doi.org/10.5281/zenodo.21112867](https://doi.org/10.5281/zenodo.21112867),
matching the paper's Data Availability statement.

## Repository structure

```
scripts/        Numbered pipeline scripts, run in order (see below)
checkpoints/    Trained GCN/GAT/MLP model weights (state_dict .pt files)
requirements.txt
README.md
CHANGELOG.md    Summary of revisions made for consistency with the paper
LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

**Raw flight data — canonical source:** U.S. Bureau of Transportation
Statistics (BTS), Reporting Carrier On-Time Performance database, 2019
(<https://www.transtats.bts.gov/>). Export the 2019 "Reporting Carrier
On-Time Performance" table as CSV and place it in the working directory.
`01_build_graphs_bts.py` reads this official BTS export directly; its
`COL_MAP` is set to the official BTS column names (`FlightDate`,
`Origin`, `Dest`, `DepDelay`, `ArrDelay`, `Cancelled`).

*Optional convenience:* because transtats.bts.gov serves data through a
form-based export rather than a direct file URL, the same BTS records are
also available via a Kaggle mirror
(`robikscube/flight-delay-dataset-20182022`, files named
`Combined_Flights_YYYY.csv`). The core column names are identical, so the
pipeline runs unchanged. This is only a download convenience; the
canonical source cited in the paper is the official BTS database.

**Weather data:** Iowa Environmental Mesonet ASOS archive, downloaded by
`04_download_weather.py`.

**Processed data files** (graph snapshots, PyG datasets — too large for
GitHub) are archived on Zenodo:
[https://doi.org/10.5281/zenodo.21113090](https://doi.org/10.5281/zenodo.21113090).

## Pipeline / how to reproduce each configuration

| Paper configuration | Scripts (in order) | Output |
|---|---|---|
| Config 1: Daily, no weather | `01_build_graphs_bts.py` → `02_prepare_pyg_temporal.py` → `03_train_temporal.py` | `graphs.pkl`, `pyg_dataset_temporal.pt`, `gcn_temporal_model.pt`, `gat_temporal_model.pt` |
| Config 2: Daily + weather | `01_build_graphs_bts.py` → `04_download_weather.py` → `05_merge_weather.py` → `02_prepare_pyg_temporal_weather.py` → `03_train_temporal_weather.py` | `graphs_weather.pkl`, `pyg_dataset_temporal_weather.pt`, `gcn_temporal_weather_model.pt`, `gat_temporal_weather_model.pt` |
| Config 3: Hourly (v1, unmasked) | `06_build_graphs_hourly.py` → `07_prepare_pyg_hourly.py` → `08_train_hourly.py` | `graphs_hourly.pkl`, `pyg_dataset_hourly.pt`, `gcn_hourly_model.pt`, `gat_hourly_model.pt` |
| Config 4: Hourly, fixed graph (unstable / NaN — z-score edge weights) | `09_build_graphs_hourly_v2.py` → `10a_prepare_pyg_hourly_v2_zscore_config4.py` → `11a_train_hourly_v2_config4_nan.py` | `pyg_dataset_hourly_v2_config4_zscore.pt`; reproduces GCN NaN failure (paper §4.4) |
| Config 5: Hourly, fixed graph (corrected) | `09_build_graphs_hourly_v2.py` → `10_prepare_pyg_hourly_v2.py` → `11_train_hourly_v2.py` | `graphs_hourly_v2.pkl`, `pyg_dataset_hourly_v2.pt`, `gcn_hourly_v2_model.pt`, `gat_hourly_v2_model.pt` |
| Table 2 / Table 3 (multi-seed, validation-based selection) | `12_multi_seed_v2.py` (reads `pyg_dataset_hourly_v2.pt`) | Test MAE mean ± std per model, 5 seeds |
| Section 4.6.3 (statistical tests) | `13_statistical_tests.py` | Paired t-test, Wilcoxon, Cohen's d |
| Config 5, **leakage-free** normalisation (Section 3.5 / 5.4) | `09_build_graphs_hourly_v2.py` → `10b_prepare_pyg_hourly_v2_trainonly.py` → `12b_multi_seed_v2_trainonly.py` | `pyg_dataset_hourly_v2_trainonly.pt`; leakage-free multi-seed Test MAE |

Config 4 and Config 5 share the same graph build (`09_build_graphs_hourly_v2.py`);
they differ only in the edge-weight normalisation applied at the PyG-preparation
step (z-score for Config 4, log1p + min-max for Config 5). Each numbered script
prints its output filename; run them in numeric order within a configuration.
Scripts write outputs to the current working directory.

## Reproducibility notes

These notes document methodological details identified during an
internal review of the code against the paper text. All items previously
flagged for pre-submission attention have been reconciled in this
revision; see `CHANGELOG.md` for the full list of changes.

1. **Batch size — consistent.** The paper (Section 3.9) and the code both
   use `batch_size=8` for all five configurations. `batch_size=8`
   reproduces the reported numbers (e.g. Config 1 GAT matches Table A1 to
   3 decimal places).

2. **MLP checkpoints — provenance.** `mlp_temporal_model.pt`,
   `mlp_temporal_weather_model.pt`, `mlp_hourly_model.pt`, and
   `mlp_hourly_v2_model.pt` in `checkpoints/` were produced by re-running
   the training procedure (same seed placement, hyperparameters, and
   data), not recovered from the authors' original run (which never saved
   the MLP weights). The resulting test MAE values are close to but not
   bit-identical to the paper (Config 1: 5.922 vs. 5.952; Config 3: 7.901
   vs. 7.892; Config 5: 9.038 vs. 8.990 — differences of 0.01–0.05 min,
   consistent with ordinary run-to-run floating-point / library-version
   variation). The GCN/GAT checkpoints were **not** regenerated — they
   remain the authors' original trained weights.

3. **Config 4 scripts — reconstructed.** The two Config 4 scripts
   (`10a_prepare_pyg_hourly_v2_zscore_config4.py`,
   `11a_train_hourly_v2_config4_nan.py`) were reconstructed from the
   Config 5 scripts to implement exactly the method described in the paper
   (Section 3.6 / 4.4): z-score edge-weight normalisation with the Config 5
   stability corrections (LayerNorm, gradient clipping, reduced GAT
   learning rate) reverted. They deterministically reproduce the
   documented failure mode (GCN → NaN; MLP and GAT run, GAT unstable).
   They save no model weights, as there is no valid GCN checkpoint to
   save for this configuration.

4. **Seed handling in single-run scripts.** In `03_train_temporal.py`,
   `08_train_hourly.py`, and `11_train_hourly_v2.py`,
   `torch.manual_seed(42)` is called once at the top, before the MLP, GCN
   and GAT are trained sequentially — so only the MLP starts from a clean
   seed-42 state; GCN and GAT inherit the RNG state left by the previous
   model. This affects only the preliminary single-run results in
   Appendix A, not the multi-seed results in Table 2
   (`12_multi_seed_v2.py` correctly reseeds before each model).

5. **Normalisation leakage — leakage-free path provided.** The
   standardisation statistics for the reported results were computed over
   the full year (Section 3.5), a mild information leakage. A leakage-free
   path fits every statistic (node-feature mean/std, edge-weight log1p
   min/max, target mean/std) on the training-visible graphs only:
   `10b_prepare_pyg_hourly_v2_trainonly.py` →
   `12b_multi_seed_v2_trainonly.py`. Train-only vs. full-year statistics
   differ by at most 3.2% (feature std), 2.3% (target std) and 3.9%
   (target mean); because standardisation is an affine transform applied
   identically to all three models, this cannot alter the between-model
   comparison (MLP test MAE shifts by under 0.1 min/seed).

## License

MIT License — see [`LICENSE`](LICENSE). (Zenodo records the same license
for the archived release.)
   Revision experiments: see revision/README_revision.md
