import json
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from pathlib import Path

from evaluation import (
    calculate_predictive_metrics,
    extract_targets,
    legacy_nacrps_and_mse,
    legacy_nacrps_statistics,
)
from forecast_store import ForecastShardStore
from metrics_stream import PredictiveMetricsAccumulator
from reproducibility import seed_everything

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername='',
    seed=2026,
):
    seed_everything(seed)
    optimizer = Adam(model.parameters(), lr=config['train']['lr'], weight_decay=5e-8)
    if foldername != '':
        output_path = foldername + '/model.pth'
    m = []
    for i in range(int(config['train']['epochs'] / 10)):
        m.append(i * 10)
        
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=m[1:], gamma=0.8)
    # earlystopping count
    ct = 0
    _, target_var = pickle.load(open('preprocess/data/var.pkl', 'rb'))
    size_y = 10 * len(target_var)
    best_valid_loss = np.inf
    for epoch_no in range(config['train']['epochs']):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch, config['diffusion']['size'], size_y)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        'avg_epoch_loss': avg_loss / batch_no,
                        'epoch': epoch_no,
                        'lr': optimizer.param_groups[0]['lr']
                    },
                    refresh=False,
                )
                
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            NACRPS_valid, _ = evaluate(
                0, model, valid_loader, nsample=5, foldername=foldername, seed=seed + 1,
                tail_thresholds=config.get('evaluation', {}).get('tail_thresholds'),
                split_name="val_model", save_shards=False, collect_metrics=False,
            )
            print('{} (best)'.format(round(best_valid_loss, 4)))
            if NACRPS_valid is None:
                raise ValueError("Validation NACRPS is undefined; the model-selection split has no valid score.")
            print('{} (current)'.format(round(NACRPS_valid, 4)))
            if best_valid_loss > NACRPS_valid:
                ct = 0
                best_valid_loss = NACRPS_valid
                torch.save(model.state_dict(), output_path)
                print('model updated')
            else:
                ct += 1
                print('count: {}'.format(ct))
            # earlystopping
            if ct > 2:
                print('stop')
                break

    if valid_loader is not None and best_valid_loss < np.inf:
        model.load_state_dict(torch.load(output_path))

def calc_metrics(is_test, all_generation, all_samples_y):
    """Return the legacy NACRPS and mean-forecast standardized MSE."""
    nacrps, mse = legacy_nacrps_and_mse(is_test, all_generation, all_samples_y)
    mse_tensor = None if mse is None else torch.as_tensor(mse, dtype=all_generation.dtype, device=all_generation.device)
    return nacrps, mse_tensor

def evaluate(is_test, model, data_loader, nsample=100, foldername="", seed=2026,
             tail_thresholds=None, split_name=None, save_shards=True, collect_metrics=True):
    generator = None
    if seed is not None:
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

    split_name = split_name or ("test" if is_test == 1 else "val_model")
    metrics = PredictiveMetricsAccumulator()
    nacrps_numerator = 0.0
    nacrps_denominator = 0.0
    nacrps_count = 0
    squared_error_sum = 0.0
    valid_count = 0
    batch_count = 0
    variable_names = target_ids = means = stds = reference_scales = reference_metadata = None
    shard_store = None

    if foldername and collect_metrics:
        with open("preprocess/data/var.pkl", "rb") as file:
            variable_names, target_ids = pickle.load(file)
        with open("preprocess/data/mean_std.pkl", "rb") as file:
            means, stds = pickle.load(file)
        reference_path = "preprocess/data/evaluation_reference_scale.pkl"
        try:
            with open(reference_path, "rb") as file:
                reference = pickle.load(file)
            reference_scales = reference["scales"]
            reference_metadata = {key: reference.get(key) for key in ("version", "seed", "source")}
        except FileNotFoundError:
            pass
        if save_shards:
            shard_store = ForecastShardStore(
                Path(foldername) / "evaluation_shards" / f"{split_name}_n{nsample}",
                {"split": split_name, "source": "TDSTF", "nsample": nsample, "seed": seed},
            )

    with torch.no_grad():
        model.eval()
        for batch in tqdm(data_loader, mininterval=5.0, maxinterval=50.0):
            generation, samples_y, samples_x = model.evaluate(batch, nsample, generator=generator)
            generation_cpu = generation.detach().cpu().numpy()
            samples_y_cpu = samples_y.detach().cpu().numpy()
            samples_x_cpu = samples_x.detach().cpu().numpy()
            info_cpu = batch["info"].detach().cpu().numpy()
            batch_count += 1
            if shard_store is not None:
                shard_store.add(generation_cpu, samples_y_cpu, samples_x_cpu, info_cpu)
            nacrps = legacy_nacrps_statistics(is_test, generation_cpu, samples_y_cpu)
            if nacrps["numerator"] is not None:
                nacrps_numerator += nacrps["numerator"]
                nacrps_denominator += nacrps["denominator"]
                nacrps_count += nacrps["count"]
            _, batch_mse = legacy_nacrps_and_mse(is_test, generation_cpu, samples_y_cpu)
            target_batch = extract_targets(samples_y_cpu)
            count = int(target_batch.mask.sum())
            if batch_mse is not None:
                squared_error_sum += batch_mse * count
                valid_count += count
            if variable_names is not None:
                report = calculate_predictive_metrics(
                    generation_cpu, samples_y_cpu, variable_names, target_ids,
                    means, stds, reference_scales=reference_scales, info=info_cpu,
                    tail_thresholds=tail_thresholds, include_patient_statistics=True,
                )
                metrics.add(report)

    if batch_count == 0:
        raise ValueError(f"Evaluation split '{split_name}' contains no batches")
    NACRPS = nacrps_numerator / nacrps_denominator if nacrps_denominator else None
    MSE = squared_error_sum / valid_count if valid_count else None
    metrics_report = metrics.finalize() if variable_names is not None else None
    nacrps_report = {
        "numerator": nacrps_numerator if nacrps_count else None,
        "denominator": nacrps_denominator,
        "count": nacrps_count,
        "value": NACRPS,
    }
    if NACRPS is None:
        nacrps_report["reason"] = "no_valid_observations" if not nacrps_count else "zero_absolute_target_sum"
    mse_report = {
        "numerator": squared_error_sum if valid_count else None,
        "denominator": valid_count,
        "count": valid_count,
        "value": MSE,
    }
    if MSE is None:
        mse_report["reason"] = "no_valid_observations"

    if foldername and collect_metrics:
        report = {
            "split": split_name,
            "population": f"{split_name} patient partition",
            "source": "TDSTF checkpoint forecasts",
            "measurement_source": "MIMIC-IV preprocessed minute-level signals; event-level provenance is not retained by the current aggregation",
            "nsample": nsample,
            "NACRPS": nacrps_report,
            "MSE_mean_standardized_legacy": mse_report,
            "evaluation_reference_scale": reference_metadata,
            "predictive_metrics": metrics_report,
            "shards": None if shard_store is None else str(shard_store.directory),
        }
        report_path = Path(foldername) / f"metrics_{split_name}_nsample{nsample}.json"
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        if is_test == 1:
            with (Path(foldername) / f"result_nsample{nsample}.pkl").open("wb") as file:
                pickle.dump([NACRPS, MSE], file)

    mse_tensor = None if MSE is None else torch.as_tensor(MSE, dtype=torch.float32)
    return NACRPS, mse_tensor
