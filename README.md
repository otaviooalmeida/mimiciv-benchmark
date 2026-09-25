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

## Temporal windows
Each sample uses **60 minutes of history to forecast the following 20 minutes**:
- History: `[start, start + 60)`; targets: `[start + 60, start + 80)`.
- Sliding windows advance by **20 minutes**, starting at ICU admission. All valid windows are retained, not just the first per stay.
- Windows do not cross ICU stays; train/validation/test remain split by patient.
- Observations retain their minute bins without interpolation. A window needs history, at least two observations of one target signal, and data reaching its final minute bin.
- Targets remain HR, SBP, DBP, Temperature and SpO2. Missing target observations are masked; the output supports 20 observations per signal.

Durations and stride are defined in `preprocess/windowing.py`. `diffusion.time_points` in `config/base.yaml` covers all 80 minutes. `diffusion.size` is the cap on selected history **observations**, not a duration.

Existing preprocessed datasets must be regenerated (step 1 may be reused):

```bash
cd preprocess
python3 step_2.py
python3 step_3.py
python3 step_4.py
cd ..
python3 main.py
```

Retrain for the new horizon; old checkpoints were trained on 30+10 minute windows.
Inference CSVs identify windows by `(sample_id, window_start)`, where `sample_id` is the ICU stay index and `window_start` is minutes since admission. Plot filenames include both identifiers.

# Experiments
Run the file main.py

To test a pretrained model, please assign the model folder name to the parameter "modelfolder"

# Tests
Run `python3 -m pytest -q` with the project dependencies and pytest installed.

# Acknowledgements
A part of the codes is based on [CSDI](https://github.com/ermongroup/CSDI) and [STraTS](https://github.com/sindhura97/STraTS)
