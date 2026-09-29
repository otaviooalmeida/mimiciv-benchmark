# TDSTF
This is the github repository for the paper "A Transformer-based Diffusion Probabilistic Model for Heart Rate and Blood Pressure Forecasting in Intensive Care Unit" (https://doi.org/10.1016/j.cmpb.2024.108060)

# MIMIC-IV data
Download the MIMIC-IV release used for the experiment (currently expected: v3.1) from

https://physionet.org/content/mimiciv/3.1/

# Environment
[Anaconda3-2023.03-0-Windows-x86_64](https://repo.anaconda.com/archive/)

[Pytorch=2.1.1 + cuda=11.8](https://pytorch.org/)

# Data preprocessing
Create empty folders `/save`, `/preprocess/data`, and `/preprocess/data/first`.
Place the MIMIC-IV release under `/preprocess/MIMICIV` with its `icu/` and `hosp/` files.
From `/preprocess`, run steps 1–4 in order (set the release label on step 1):

```bash
MIMIC_DATA_VERSION=3.1 python step_1.py
python step_2.py
python step_3.py
python step_4.py --seed 2026
```

# Auditable events and causal availability

Preprocessing step 1 writes `preprocess/data/clinical_events.csv` and
`clinical_event_audit.json`, preserving IDs, source dictionaries, raw values/units,
measurement and availability times, and inputevent rate/amount/order/status metadata.
The item IDs follow the existing feature mappings; MIT-LCP SQL is pinned to commit
`303d26c623dcc9c49cc0f204468d4acc2f063797` as guidance, not universal clinical truth.
Set `MIMIC_DATA_VERSION` before step 1 to record the exact release label. Details and
limitations: [`docs/data/clinical-events.md`](docs/data/clinical-events.md).

The target is now peripheral `SpO2_peripheral`; lab `SO2_bloodgas` remains separate.
Step 2 excludes missing `storetime` by default and supports an explicit sensitivity run:
`python step_2.py --missing-availability-policy measurement_time`. Step 3 also enforces
measurement time before cutoff and availability at/before cutoff. Retrospectively expanded
inputevent medication features are excluded until an as-of infusion state can be reconstructed.
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
population and model/baseline source. `clinical_events.csv` preserves event provenance;
the current forecast shards still use the legacy minute aggregation and do not carry a
source ID for each model input/target.

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
