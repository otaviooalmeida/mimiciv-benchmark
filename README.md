# TDSTF
This is the github repository for the paper "A Transformer-based Diffusion Probabilistic Model for Heart Rate and Blood Pressure Forecasting in Intensive Care Unit" (https://doi.org/10.1016/j.cmpb.2024.108060)

# MIMIC-IV data
Download the MIMIC-IV release used for the experiment (currently expected: v3.1) from

https://physionet.org/content/mimiciv/3.1/

# Environment
[Anaconda3-2023.03-0-Windows-x86_64](https://repo.anaconda.com/archive/)

[Pytorch=2.1.1 + cuda=11.8](https://pytorch.org/)

# Data preprocessing
Place the MIMIC-IV release under `/preprocess/MIMICIV` with its `icu/` and `hosp/` files.
For the audited clinical-event dataset, install the Parquet dependencies and build from
chunks (run from `/preprocess`):

```bash
pip install -r requirements-events.txt
MIMIC_DATA_VERSION=3.1 python build_event_dataset.py --mimic-root MIMICIV --output-dir data
```

This writes local, ID-bearing Parquet event/cohort files, minute/source aggregates, an
ID-free cohort-flow JSON, a variable/source summary and a stratified extreme-value review
sample. Do not commit generated data. This audit build is separate from the forecast
preprocessing below; it is not silently substituted for the legacy forecast protocol.

The legacy forecast preprocessing is also chunked and disk-backed. From `preprocess/`:

```bash
pip install -r requirements-events.txt
python step_1.py --chunksize 250000
python step_2.py --chunksize 250000 --staging-buckets 64 \
  --max-staging-bucket-rows 2000000 --max-stay-rows 2000000
python step_3.py --workers 1
python step_4.py --seed 2026
```

Step 1 writes mapped rows incrementally; step 2 writes complete-stay Parquet partitions;
step 3 sends file paths (not DataFrames) to spawn-safe workers; step 4 fits train-only
statistics online and writes sharded data consumed lazily by `dataset.py`. Step 2 stages into a fixed number of hash buckets to avoid a file per stay per input
chunk. Its stay and staging-bucket row limits are safety guards, not external spill:
if either trips, stop and investigate rather than raising limits blindly. Start with one step-3
worker, then increase only after measuring process-tree memory on the authorized machine.
The audit Parquet builder remains the path that writes the raw-provenance event table and
clinical audit report.

# Auditable events and causal availability

The chunked `build_event_dataset.py` writes `events.parquet/`, `cohort.parquet`,
`minute_aggregates.parquet/`, `cohort_flow.json`, `variable_source_summary.parquet`, and
`extreme_review_sample.parquet`. Coverage is limited to the existing validated benchmark
item mappings (primary targets, named supporting variables, selected outputs, and mapped
norepinephrine/vasopressin input events), not a full MIMIC table dump. It validates ICU temporal membership, uses half-open
`[intime, outtime)` boundaries, links labs with missing `hadm_id` only when patient/time
identifies one stay, and retains ambiguous/unattributable rows without duplicating them.
Exact duplicates use a documented key and preserve counts/IDs; simultaneous repeats and
cross-source discordance remain explicit. Details: [`docs/data/clinical-events.md`](docs/data/clinical-events.md).

Peripheral `SpO2_peripheral` and lab `SO2_bloodgas` remain distinct signals.
The fixed primary minute statistic is the arithmetic mean per variable/source/unit; sources
are never pooled, and min/max/last are auxiliary. No target winsorization is applied.
Legacy step 1 also no longer applies its historical value-range filters to mapped values.
The auditable Parquet path records reasons for invalid units and physically impossible
values while retaining plausible extremes with review flags. Step 2 in the legacy pipeline
excludes missing `storetime` by default and supports an explicit sensitivity run:
`python step_2.py --missing-availability-policy measurement_time`. Step 3 also enforces
measurement time before cutoff and availability at/before cutoff. Retrospectively expanded
inputevent medication features are excluded until an as-of infusion state can be reconstructed. The legacy forecast shards are not yet built from the new source-specific Parquet aggregates.
Regenerate preprocessing steps 1–4 after this target/protocol change. If the old frozen
scale exists, archive `preprocess/data/evaluation_reference_scale.pkl` explicitly before
step 4; step 4 refuses to silently reuse or overwrite a scale for the former mixed target.

# Experiments

`main.py` trains and evaluates on `val_model` by default; use `--split calibration` for
calibration reports. The original 16% validation partition is split by patient into
model-selection and calibration populations (approximately 8% each), while the 20% test
partition remains untouched unless explicitly requested. The normalizer is fit on training
patients only; validation and calibration patients do not determine its statistics.

```bash
python main.py --seed 2026 --split val_model
python main.py --modelfolder <run> --split test
```

Standalone inference defaults to `val_model`; specify `--split calibration` or
`--split test` to select another population:

```bash
python infer.py --checkpoint save/<run>/model.pth --split test
```

# Reproducibility

The default seed is `2026` (`seed` in `config/base.yaml`). To regenerate and record
patient-level splits, run preprocessing step 4 from the `preprocess` directory:

```bash
python step_4.py --seed 2026
```

Train/evaluate with the same seed:

```bash
python main.py --seed 2026
```

Standalone inference uses the same sampling seed by default. Override it explicitly
when desired; `--data-seed` can reproduce the dataset's deterministic context selection
independently of the diffusion sampling seed:

```bash
python infer.py --checkpoint save/<run>/model.pth --seed 2026
```

Runs save their seed/configuration in `run_metadata.json`, and step 4 saves split
membership and its seed in `preprocess/data/splits.pkl`. `main.py` rejects a recorded
split seed that differs from `--seed`. For old datasets without split metadata, it warns;
rerun step 4 to make the split itself reproducible. Deterministic PyTorch algorithms are
enabled, which can reduce performance or report an unsupported nondeterministic kernel.
Reproduction is intended for the same data, software versions, device, and hardware; exact
results across different hardware/CUDA versions are not guaranteed.

# Evaluation metrics

Evaluation writes `metrics_*.json`, `baseline_metrics.json`, batch-aligned forecast shards
(`generation`, targets, history, and patient/sample metadata), and prediction/calibration
plots. It does not concatenate the full test set on GPU. Reports identify the split,
population and model/baseline source. The new `events.parquet` dataset preserves event
provenance; legacy forecast shards still use the older minute aggregation and do not carry
a source ID for each model input/target.

Metrics include mean-based MSE, median-based MAE, empirical and fair ensemble CRPS,
80%/95% coverage, width and interval score, plus upper/lower threshold-weighted CRPS.
Tail thresholds are explicit in `config/base.yaml` and scores include all valid queries.
Per-signal values use original clinical units; cross-signal micro, macro-by-variable,
and patient means use frozen per-signal scales. Numerators, denominators, counts and
undefined-score reasons are included. The frozen reference is
`preprocess/data/evaluation_reference_scale.pkl`; retain it when changing the normalizer.

`baseline_metrics.json` includes training-mean, persistence, causal regularized trend,
moving-average, training-prevalence risk and a small causal logistic risk classifier.
Missing history falls back to training means and reports fallback counts and time since
last measurement.

`NACRPS` retains its legacy quantile-grid formula and normalization by the sum of
absolute standardized targets. It is reported for continuity, not as a score comparable
across tasks or cohorts. Empty masks and zero denominators produce null scores with a
count/reason (or a protocol error for an empty evaluation split), never a fabricated zero.

# Acknowledgements
A part of the codes is based on [CSDI](https://github.com/ermongroup/CSDI) and [STraTS](https://github.com/sindhura97/STraTS)
