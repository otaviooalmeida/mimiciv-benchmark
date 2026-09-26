# TDSTF
This is the github repository for the paper "A Transformer-based Diffusion Probabilistic Model for Heart Rate and Blood Pressure Forecasting in Intensive Care Unit" (https://doi.org/10.1016/j.cmpb.2024.108060)

# MIMIC-IV data
Download MIMIC-IV 3.1 at

https://physionet.org/content/mimiciv/3.1/

# Environment
[Anaconda3-2023.03-0-Windows-x86_64](https://repo.anaconda.com/archive/)

[Pytorch=2.1.1 + cuda=11.8](https://pytorch.org/)

# Data preprocessing
Create `save/`, `preprocess/data/`, and `preprocess/data/first/`.
Place the extracted MIMIC-IV CSVs in `preprocess/MIMICIV/` (`icu/` and `hosp/`).
Run `step_1.py` through `step_4.py` in order from `preprocess/`.

## Disk-backed preprocessing and datasets

Steps 3/4 and the training DataLoader no longer accumulate all windows or all
normalized tensors in RAM. Pandas, NumPy and PyTorch remain in use; no new
third-party storage dependency is required.

- **Step 2:** stores events once in read-only mmap-compatible arrays under
  `data/events_*/`. `data/sets.pkl` is a small versioned manifest, not a list of
  DataFrames. Per-stay ordering retains the previous Pandas sorting semantics.
- **Step 3:** reads only the event ranges needed for each 30+10-minute window,
  even for long stays. It writes eight int64 metadata/offset fields per accepted
  window, not copies of the observations. `data/first/manifest.pkl` identifies
  the completed `data/windows_*/` snapshot. Workers receive offsets, not tables.
- **Step 4:** groups window indices with an on-disk SQLite index (Python standard
  library). Temporary per-variable files replace the old in-memory `rec` lists.
  Statistics are reduced in NumPy-sized buffers. It saves split indices under
  `data/dataset_*/`, a small `data/dataset.pkl` manifest and `data/mean_std.pkl`.
- **Dataset:** `get_dataloader()` reads the manifest automatically. Each sample
  reads its event ranges, normalizes them and applies the same history selection
  and masks as before. No whole-cohort `samples_x`/`samples_y` allocation occurs.

For an existing installation that exhausted RAM, **rerun the updated step 2**
(step 1 output can be reused):

```bash
cd preprocess
python3 step_2.py
python3 step_3.py --workers 1 --chunk-size 1000
python3 step_4.py
cd ..
python3 main.py
```

`--chunk-size` now limits buffered **index rows**, not observations or the number
of retained windows. `--workers` defaults to 1 and limits pending tasks as well
as running workers. Increase it only after measuring memory on your machine.
After a failed step 3, rerun it to completion; step 4 rejects its incomplete
snapshot rather than silently using partial outputs.

**Benchmark semantics are preserved:** all eligible windows, required/optional
signals, missing-value masks, history selection and 64/16/20 patient split remain
unchanged. Normalization still uses **training + validation histories**, counts
observations again for each overlapping window, uses population std (`ddof=0`),
and retains the old zero-std/unseen-variable handling. Buffer-sized reductions
preserve NumPy's rounding order, including constant/near-constant signals; parity
tests compare original and disk-backed tensors and statistics exactly in the
same NumPy environment. Defaults remain unseeded; `step_4.py --seed 2026` is an
optional reproducible patient split, not a new splitting policy. Unseeded runs,
changed numerical-library versions or nondeterministic GPU execution are not a
promise of bit-for-bit training reproducibility.

**Compatibility and disk space:**

- Old `sets.pkl` files are **not** automatically loaded/converted by step 3.
  `--legacy-input` explicitly enables conversion when the old input fits RAM;
  it is not a low-memory workaround. Prefer regenerating it with step 2.
- Step 4 can consume already-completed legacy `samples_*.pkl` files one at a
  time. That compatibility path still needs enough RAM for the largest old
  chunk. Legacy normalized `dataset.pkl` lists remain readable by the loader,
  but only the new manifest path provides disk-backed sample loading.
- Keep the **whole `data/` directory**, not just its pickle manifests. Paths
  between snapshots are relative. Completed snapshots are published atomically
  and retained on reruns so existing datasets do not change underneath training.
  Remove old snapshots only after confirming no dataset/window manifest refers
  to them. Do not mix variable metadata from different preprocessing runs.
- Step 4 needs temporary disk space for its index and **8 bytes per historical
  observation used in normalization, counting window repetitions**. These
  temporary statistics and the SQLite index are removed after a successful run.
  New window/event snapshots avoid permanent duplication of overlapping events.

These changes address step 3, step 4 and input loading. Steps 1/2 still have
in-memory Pandas operations; the diffusion model and final evaluation still have
their own memory requirements (evaluation currently accumulates predictions).
Memory also depends on the observations within a single window, patient-ID
metadata, DataLoader shuffle indices and the number of workers. Run the RSS
regression with `python3 -m pytest -q -s tests/test_disk_memory.py` on Linux.

## Temporal windows
Each sample uses **30 minutes of history to forecast the following 10 minutes**:
- History: `[start, start + 30)`; targets: `[start + 30, start + 40)`.
- Sliding windows advance by **10 minutes**, starting at ICU admission. All valid windows are retained, not just the first per stay.
- Windows do not cross ICU stays; train/validation/test remain split by patient.
- Observations retain their minute bins without interpolation. The last recorded minute of the stay must reach or exceed the final minute bin of a candidate window; coverage within it is checked separately below.
- Targets are **HR, SBP, RR, Temperature and SpO2**. **HR, SBP and RR are required**; Temperature and SpO2 are optional within each window. DBP remains available as historical context, never as a future target. Missing target observations are masked; the output supports 10 observations per signal.

Durations and stride are defined in `preprocess/windowing.py`. `diffusion.time_points: 40` in `config/base.yaml` covers all 40 minutes. `diffusion.size` is the cap on selected history **observations**, not a duration.

Existing preprocessed datasets must be regenerated for the 30+10-minute horizon, new targets and stricter coverage (step 1 may be reused because it already extracts RR):

```bash
cd preprocess
python3 step_2.py
python3 step_3.py
python3 step_4.py
cd ..
python3 main.py
```

Retrain after changing targets or window eligibility. Do not reuse DBP-target checkpoints to evaluate RR, even if their tensor shapes match. Checkpoints trained on 60+20-minute windows also require retraining for the current 30+10-minute horizon.
Inference CSVs identify windows by `(sample_id, window_start)`, where `sample_id` is the ICU stay index and `window_start` is minutes since admission. Plot filenames include both identifiers.

## Window quality: required HR, SBP and RR coverage

Before history subsampling, `preprocess/step_3.py` checks **each** of HR, SBP and RR independently:

| Rule | Default |
| --- | --- |
| History observation count | At least 4 distinct observed minute bins per signal |
| History distribution | At least one observation in each block: `[0,10)`, `[10,20)`, `[20,30)` |
| History recency | Last observation at most 5 minutes before forecast onset: minute 25 or later |
| History internal gaps | At most 10 minutes between consecutive observations of the same signal |
| Future observation count | At least 2 distinct observed minute bins per signal |
| Future distribution | At least one observation in each half: `[30,35)`, `[35,40)` |

Times are relative to each window. Maximum ages and gaps are inclusive. A failure for **any required signal** rejects the whole window. Nonfinite values in the history or retained future targets also reject it. Temperature and SpO2 do not need minimum coverage; their available observations remain in the sample. DBP does not affect signal-coverage eligibility.

Thresholds are initial research settings, **not clinically validated criteria**. Configure them in `config/windowing.yaml`, or run `python3 step_3.py --quality-config /path/to/windowing.yaml` from `preprocess/`. Omitted settings use the documented defaults. Tune using training/validation data, not test outcomes. Future coverage is an offline label-availability criterion, not an eligibility rule usable at forecast time; no selection depends on future changes or event labels. Two future observations do not establish absence of events between measurements.

`preprocess/data/window_quality_report.json` records the horizon/stride, effective thresholds, accepted/rejected window counts, unique patients retained/lost, and rejection counts by rule. A window can fail multiple rules, so reason counts overlap. Candidates are only the 40-minute windows reached by a stay's last record; patients without any complete candidate are still counted among patients lost. Rerunning step 3 publishes a new window-index snapshot and removes legacy `samples_*.pkl` outputs to avoid mixing cohorts. Inspect the report before step 4, especially if the stricter profile retains few patients.

## History selection: guaranteed recent target observations

`dataset.py` uses the same deterministic selection for training, validation and inference:

1. Reserve up to the **three latest available observations of each target signal**, ordered by timestamp. With five targets, this uses at most 15 of the 60 input positions.
2. Allocate the remaining positions to the least-represented variables, including non-target covariates. Ties favor the variable with the most recent remaining observation, then its ID.
3. Within each variable's context quota, choose evenly spaced observation indices, including the oldest and latest remaining observations when at least two slots are available. A single context slot takes the latest remaining observation.

Configure the cap and reservation in `config/base.yaml`:

```yaml
diffusion:
  size: 60
  recent_per_target: 3
```

Both settings must be positive integers. If a window cannot fit all reserved observations within `size`, loading fails with instructions to increase the budget or reduce the reservation; the guarantee is never silently discarded. Sparse histories retain all available observations and use masked padding, without interpolation or invented measurements. "Latest" means latest **available within the 30-minute history**, not necessarily recently measured if a signal is sparse. Non-target context coverage remains best-effort under the budget.

Selection only inspects historical variable IDs and timestamps, never future targets or values. It preserves the original sample arrays and future targets, and returns the selected history chronologically.

History selection happens in the DataLoader; changing only its budget/reservation does not require preprocessing again, but changing targets or window-quality rules **does**. Old checkpoints may remain structurally loadable but must be retrained to evaluate changed targets, eligibility or conditioning fairly. The history-selection policy and budget are recorded in inference `metrics.json`. Window coverage is checked before this observation cap; the cap does not guarantee preservation of every coverage block. This guarantees access to the latest available measurements, not successful prediction of future peaks or drops.

# Experiments
Run the file main.py

To test a pretrained model, please assign the model folder name to the parameter "modelfolder"

## Inference: trajectories and event probabilities

```bash
python3 infer.py --checkpoint save/EXPERIMENT/model.pth \
  --nsample 100 --n-examples 5 --n-trajectories 20 --seed 2026
```

Outputs retain the existing CSV, NPZ and violin plots, and add:

- `forecast_trajectories_*_window_*.png`: reproducibly selected complete diffusion draws, actual observations, pointwise median and marginal 95% intervals. The same draw indices are used across times and signals; selection never depends on the future ground truth. `--n-trajectories` affects visualization only and is capped at `--nsample`.
- `signal_distribution_violin.png`: compares actual values, pointwise predictive medians, and samples from the full predictive distribution. Each group uses at most 10,000 uniformly sampled values per signal to bound plotting cost. These are **marginal** comparisons, not evidence of temporal accuracy or conditional calibration.

To also estimate abrupt-rise/drop probabilities, explicitly supply per-signal thresholds:

```bash
python3 infer.py --checkpoint save/EXPERIMENT/model.pth \
  --nsample 100 --n-trajectories 20 --event-config config/events.example.yaml
```

**The example thresholds are illustrative, not validated clinical alarm criteria.** Copy and adapt them to the study using training/validation data, not test outcomes. Rules specify `min_change` in original signal units and `max_gap_minutes`. An event is a change of at least that magnitude between consecutive observations of the same signal, within the maximum gap; a rise is not necessarily a confirmed local peak.

With `--event-config`, `event_probabilities.csv` reports the fraction of **all** generated trajectories containing at least one rise/drop for each configured signal and window. A trajectory can contain both. The latest available observation in the model's selected history can anchor the first future observation. The CSV also records observed event flags, the number of eligible pairs/draws and the thresholds; the probabilities appear on trajectory plots. With no eligible pairs, probabilities and observed flags are blank, not zero. Without an event config, event estimation is disabled.

Events are evaluated only on the dataset's observed future query grid. Missing observations and large gaps do not establish absence of an event; connected plot lines do not reconstruct unobserved physiology. The marginal median can remain flat even when individual trajectories contain drops at different times. Finite-sample probabilities are model estimates, not automatically calibrated risks.

`metrics.json` records the seed, history-selection policy, plotted trajectory count and event rules; existing SACRPS/MSE calculations are unchanged. The plotting/event diagnostics themselves do not require retraining; changing the history-selection policy should be evaluated with retraining as described above.

# Tests
Run `python3 -m pytest -q` with the project dependencies and pytest installed.

# Acknowledgements
A part of the codes is based on [CSDI](https://github.com/ermongroup/CSDI) and [STraTS](https://github.com/sindhura97/STraTS)
