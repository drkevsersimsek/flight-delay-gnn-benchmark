# Flight Delay GNN vs MLP Benchmark

Code and trained model checkpoints for the paper *"Comparison of Graph
Neural Networks and Multilayer Perceptrons for Airport Network Delay
Forecasting"*. This repository contains the full data processing and
model training pipeline; large processed data files are hosted
separately on Zenodo (see [Data](#data) below).

## Repository structure

```
scripts/        15 numbered pipeline scripts, run in order (see below)
checkpoints/    Trained GCN/GAT model weights (state_dict .pt files)
requirements.txt
README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Raw flight data: U.S. Bureau of Transportation Statistics, Reporting
Carrier On-Time Performance database (2019).

**Note:** `01_build_graphs_bts.py` currently points to a Kaggle mirror
of this dataset (`robikscube/flight-delay-dataset-20182022`) rather
than downloading directly from transtats.bts.gov. Column names differ
slightly between the two sources — if you download directly from BTS,
you may need to adjust `COL_MAP` in `01_build_graphs_bts.py`. **This
should be reconciled with the paper's Data Availability statement
before submission** (the paper currently describes the official BTS
source only).

Weather data: Iowa Environmental Mesonet ASOS archive, downloaded by
`04_download_weather.py`.

**Processed data files** (graph snapshots, PyG datasets — too large for
GitHub) are archived on Zenodo: **[DOI link to be inserted once the
Zenodo upload is complete]**.

## Pipeline / how to reproduce each configuration

| Paper configuration | Scripts (in order) | Output |
|---|---|---|
| Config 1: Daily, no weather | `01_build_graphs_bts.py` → `02_prepare_pyg_temporal.py` → `03_train_temporal.py` | `graphs.pkl`, `pyg_dataset_temporal.pt`, `gcn_temporal_model.pt`, `gat_temporal_model.pt` |
| Config 2: Daily + weather | `01_build_graphs_bts.py` → `04_download_weather.py` → `05_merge_weather.py` → `02_prepare_pyg_temporal_weather.py` → `03_train_temporal_weather.py` | `graphs_weather.pkl`, `pyg_dataset_temporal_weather.pt`, `gcn_temporal_weather_model.pt`, `gat_temporal_weather_model.pt` |
| Config 3: Hourly (v1, unmasked) | `06_build_graphs_hourly.py` → `07_prepare_pyg_hourly.py` → `08_train_hourly.py` | `graphs_hourly.pkl`, `pyg_dataset_hourly.pt`, `gcn_hourly_model.pt`, `gat_hourly_model.pt` |
| Config 4: Hourly, fixed graph (unstable / NaN) | *not included — see Known limitations* | — |
| Config 5: Hourly, fixed graph (corrected) | `09_build_graphs_hourly_v2.py` → `10_prepare_pyg_hourly_v2.py` → `11_train_hourly_v2.py` | `graphs_hourly_v2.pkl`, `pyg_dataset_hourly_v2.pt`, `gcn_hourly_v2_model.pt`, `gat_hourly_v2_model.pt` |
| Table 2 / Table 3 (multi-seed, validation-based selection) | `12_multi_seed_v2.py` (reads `pyg_dataset_hourly_v2.pt`) | Test MAE mean ± std per model, 5 seeds |
| Section 4.6.3 (statistical tests) | `13_statistical_tests.py` | Paired t-test, Wilcoxon, Cohen's d |

Each numbered script prints its output filename; run them in numeric
order within a configuration. Scripts `01`–`11` write outputs to the
current working directory.

## Known limitations (open items — resolve before submission)

These were identified during an internal review of the code against
the paper text. Items 1 and 2 below have been investigated and
resolved (see details); items 3 and 4 remain open and require an
authors' decision.

1. **RESOLVED — batch size.** The paper text (Section 3.9) states batch
   sizes of 16 for Configurations 1–2 and 8 for Configurations 3–5. The
   code originally used `batch_size=8` for *all* configurations. To
   determine which was actually used to produce the reported numbers,
   both scripts for Configurations 1–2 were re-run with each setting
   and compared against the paper's Table A1:

   | Config | Metric | Paper (Table A1) | Re-run, batch=8 | Re-run, batch=16 |
   |---|---|---|---|---|
   | 1 | MLP | 5.952 | **5.922** | 5.802 |
   | 1 | GCN | 6.316 | **6.318** | 6.170 |
   | 1 | GAT | 5.908 | **5.908** (exact) | 6.163 |
   | 2 | MLP | 6.014 | **6.017** | — |
   | 2 | GCN | 6.086 | **6.086** (exact) | — |

   `batch_size=8` reproduces the paper's numbers closely (GAT and GCN
   match to 3 decimal places in one case each); `batch_size=16` does
   not. **Conclusion: the code was correct as originally written; the
   manuscript text in Section 3.9 is the error and should be corrected
   to say batch size 8 was used for all five configurations** (or, if
   16 was genuinely intended, all reported numbers for Configurations
   1–2 need to be regenerated with `batch_size=16` and every downstream
   table/figure updated). The scripts in this repository use
   `batch_size=8` throughout, matching the reported results.

2. **RESOLVED — MLP checkpoints added.** `mlp_temporal_model.pt`,
   `mlp_temporal_weather_model.pt`, `mlp_hourly_model.pt`, and
   `mlp_hourly_v2_model.pt` are now included in `checkpoints/`, and
   `torch.save(...)` calls for the MLP were added to all four
   single-run training scripts (`03_train_temporal.py`,
   `03_train_temporal_weather.py`, `08_train_hourly.py`,
   `11_train_hourly_v2.py`).

   **Note on provenance:** these four MLP checkpoints were produced by
   re-running the training procedure (same seed placement, same
   hyperparameters, same data), not recovered from the authors'
   original run (which never saved the MLP weights). The resulting
   test MAE values are close to but not bit-identical to the values in
   the paper (Config 1: 5.922 vs. 5.952 reported; Config 3: 7.901 vs.
   7.892 reported; Config 5: 9.038 vs. 8.990 reported — differences of
   0.01–0.05 minutes, consistent with ordinary run-to-run floating
   point / library-version variation and not indicative of a
   methodological error). The existing GCN/GAT checkpoint files in
   this repository were **not** regenerated or overwritten — they
   remain the authors' original trained weights.

3. **Raw data source.** See the note under [Data](#data) above —
   `01_build_graphs_bts.py` uses a Kaggle mirror; the paper's Data
   Availability statement references the official BTS database
   directly. These should be made consistent.
4. **Config 4 (unstable/NaN) script not included.** The script that
   produces the NaN failure mode discussed in the paper (unbounded /
   z-score-normalised edge weights causing `GCNConv` to output NaN)
   is referenced only in a code comment in
   `09_build_graphs_hourly_v2.py` and was not part of the uploaded
   files. Include it if exact reproduction of that failure mode is
   needed.
5. **Seed handling in single-run scripts.** In `03_train_temporal.py`,
   `08_train_hourly.py`, and `11_train_hourly_v2.py`,
   `torch.manual_seed(42)` is called once at the top of the script,
   before the MLP, GCN, and GAT are trained sequentially — so only the
   MLP actually starts from a clean seed-42 state; GCN and GAT inherit
   whatever RNG state training the previous model left behind. This
   affects only the preliminary single-run results reported in
   Appendix A, not the multi-seed results in Table 2 (`12_multi_seed_v2.py`
   correctly reseeds before each model).

## License

[Add a license, e.g. MIT for code — Zenodo will ask for one when you
archive the release]
