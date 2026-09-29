# TDSTF
This is the github repository for the paper "A Transformer-based Diffusion Probabilistic Model for Heart Rate and Blood Pressure Forecasting in Intensive Care Unit" (https://doi.org/10.1016/j.cmpb.2024.108060)

# MIMIC-III data
Download the dataset at

https://physionet.org/content/mimiciii/1.4/

# Environment
[Anaconda3-2023.03-0-Windows-x86_64](https://repo.anaconda.com/archive/)

[Pytorch=2.1.1 + cuda=11.8](https://pytorch.org/)

# Data preprocessing
Create empty folders: "/save", "/preprocess/data", "/preprocess/data/MIMICIII", and "/preprocess/data/first"

Download the MIMIC-III data to "/preprocess/data/MIMICIII"

Run the files step_1.py through step_4.py in order in the folder "/preprocess"

# Experiments
Run the file main.py

To test a pretrained model, please assign the model folder name to the parameter "modelfolder"

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

# Acknowledgements
A part of the codes is based on [CSDI](https://github.com/ermongroup/CSDI) and [STraTS](https://github.com/sindhura97/STraTS)
