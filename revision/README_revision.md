# Revision experiments (Aeronautical Journal, revised manuscript)

All results in the revised manuscript are produced by the scripts in this folder.
Data files (`*.npz`, `*.pkl`) are on Zenodo (see the main README); place them in this folder before running.

| Manuscript result | Data | Run | Summarise |
|---|---|---|---|
| Table 2, Fig. 1, Sections 4.1–4.3 (main, graph controls, horizons, rolling origin, lagged features) | `graphs_hourly_v2.npz` | `python run_all.py` (uses `queue.txt`) and `python naive.py` | `python analyze.py` |
| Section 3.5, Table 1 row 5c (calendar-hour grid) | `graphs_hourly_cal.npz` | `python naive.py graphs_hourly_cal.npz _cal`, `python run_cal.py` | `python analyze_cal.py` |
| Section 4.3, Table 1 row 5s (sparse/directed graphs) | `graphs_hourly_v2_top5.npz`, `graphs_hourly_v2_top10.npz` | `python run_queue.py queue_sparse.txt` | `python analyze_extra.py` |
| Section 4.3, Table 1 row 5w (hourly weather) | `graphs_hourly_v2_weather.npz` | `python run_queue.py queue_weather.txt` | `python analyze_extra.py` |
| Section 3.8 (hyperparameter search and tuned five-seed re-run) | `graphs_hourly_v2.npz` | `exp.py --hidden {32,64,128} --lr {0.001,0.005} --seed 42 --out hp.jsonl`; `python run_queue.py queue_tuned.txt` | `python analyze_tuned.py` |

Building the data files from the Zenodo archive:
- `convert_pkl_to_npz.py`: `graphs_hourly_v2.pkl` → `graphs_hourly_v2.npz` (needs NumPy ≥ 2)
- `make_calendar_grid.py`: `graphs_hourly.pkl` + `graphs_hourly_v2.pkl` → `graphs_hourly_cal.npz`
- `make_sparse_graphs.py`: `graphs_hourly_v2.npz` → `graphs_hourly_v2_top5.npz`, `graphs_hourly_v2_top10.npz`
- `04b_download_weather_hourly.py`, `05c_download_hnl.py`, `05b_merge_weather_hourly.py`: hourly ASOS weather → `graphs_hourly_v2_weather.npz`

`exp.py` trains one model (MLP, GCN, GAT, GraphSAGE-style `sage`, MLP + neighbour mean `mlpnb`) for one seed with train-only
normalisation, per-epoch validation and early stopping (patience 15), and appends one line to the results file.
Tested with Python 3.10, PyTorch 2.3.1, PyTorch Geometric 2.8.1 and NumPy 1.26.

The `results/` folder holds the raw outputs used in the manuscript.
