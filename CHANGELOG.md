# Changelog — reconciliation with the revised manuscript

This revision aligns the repository with the revised manuscript
*"When Does Graph Structure Help in Airport Delay Forecasting?
A Controlled GNN–MLP Benchmark."* Each change below records what was
modified and why. No existing GCN/GAT checkpoints, and no numerical
results, were altered.

## 1. Zenodo DOIs inserted (was a placeholder)
- **Before:** README said *"[DOI link to be inserted once the Zenodo
  upload is complete]"* and had no archive DOI.
- **After:** README now cites the two DOIs used in the paper's Data
  Availability statement:
  - Code / repository archive: `10.5281/zenodo.21112867`
  - Processed data (graph snapshots, PyG datasets): `10.5281/zenodo.21113090`
- **Action still required by the authors:** confirm that the Zenodo
  uploads for these two DOIs are actually published and that their
  contents match this repository and the processed-data files. This
  edit makes the README consistent with the manuscript, but only the
  authors can verify the Zenodo depositions themselves exist.

## 2. Raw-data source reconciled to official BTS
- **Before:** `01_build_graphs_bts.py` documented a Kaggle mirror as the
  data source, while the paper's Data Availability references the
  official BTS database; the README flagged this to "be reconciled
  before submission."
- **After:** the official BTS Reporting Carrier On-Time Performance
  database (transtats.bts.gov) is documented as the canonical/primary
  source in both `01_build_graphs_bts.py` and the README, matching the
  paper. The Kaggle mirror is retained only as an explicitly optional
  download convenience providing identical BTS records. `COL_MAP` is
  unchanged (the official BTS and Kaggle core column names are the same:
  `FlightDate`, `Origin`, `Dest`, `DepDelay`, `ArrDelay`, `Cancelled`).

## 3. Missing Config 4 scripts reconstructed
- **Before:** the Config 4 (z-score edge weights → NaN) script was not
  in the repository; it was referenced only in a code comment, and the
  README pipeline table listed Config 4 as *"not included."*
- **After:** two scripts were added, reconstructed from the Config 5
  scripts to implement exactly the method described in the paper
  (Section 3.6 and 4.4):
  - `10a_prepare_pyg_hourly_v2_zscore_config4.py` — z-score edge-weight
    normalisation (produces negative edge weights).
  - `11a_train_hourly_v2_config4_nan.py` — Config 5 stability
    corrections reverted (no LayerNorm, no gradient clipping, GAT
    lr = 0.005), reproducing the documented GCN NaN failure.
- These are **reconstructions**, clearly labelled as such in their file
  headers; they are not the authors' original files. They reproduce a
  documented *failure mode* (GCN → NaN; MLP and GAT run), so no numeric
  result is fabricated. They save no model weights.

## 4. Batch size — already consistent (no change needed)
- An earlier manuscript draft stated batch sizes of 16 (Configs 1–2) and
  8 (Configs 3–5), conflicting with the code's uniform `batch_size=8`.
- The **revised manuscript (Section 3.9) now states `batch_size=8` for
  all five configurations**, which matches the code and reproduces the
  reported numbers. No code change was made. The stale "open item" note
  was removed from the README.

## 5. License finalised
- **Before:** README's License section said *"[Add a license …]"* even
  though a MIT `LICENSE` file was present.
- **After:** README now states the MIT License and links to `LICENSE`.

## 6. "Open items — resolve before submission" section reframed
- The README section previously titled *"Known limitations (open items —
  resolve before submission)"* is now **"Reproducibility notes."** The
  resolved documentation issues were folded into this changelog; the
  remaining entries are genuine, honest methodological notes (MLP
  checkpoint provenance, single-run seed handling, normalisation-leakage
  path) that are retained deliberately for transparency, not deleted.

---

## Items that require author judgement (not auto-resolved)
- **Zenodo depositions:** verify the two DOIs above are published and
  their contents match (see item 1).
- **MLP checkpoint provenance (Reproducibility note 2):** the released
  MLP weights are re-runs, not the original run. If bit-exact
  reproduction of the paper's MLP numbers is required, regenerate and
  re-report, or state the re-run provenance in the paper.
- **Confirmatory leakage-free re-run (Reproducibility note 5):** running
  `12b_multi_seed_v2_trainonly.py` for a full 3-model, 5-seed Config 5
  pass is recommended as a final check before submission.
