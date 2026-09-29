import json
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

from evaluation import calculate_predictive_metrics, legacy_nacrps_and_mse, legacy_nacrps_statistics
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
                0, model, valid_loader, nsample=5, foldername=foldername, seed=seed + 1
            )
            print('{} (best)'.format(round(best_valid_loss, 4)))
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
    mse_tensor = torch.as_tensor(mse, dtype=all_generation.dtype, device=all_generation.device)
    return nacrps, mse_tensor

def evaluate(is_test, model, data_loader, nsample=100, foldername="", seed=2026):
    generator = None
    if seed is not None:
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        model.eval()
        all_samples_x = []
        all_samples_y = []
        all_generation = []
        all_info = []
        with tqdm(data_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                # ground truth values will be replaced with pure noise before generation
                output = model.evaluate(batch, nsample, generator=generator)
                generation, samples_y, samples_x = output
                all_generation.append(generation)
                all_samples_x.append(samples_x)
                all_samples_y.append(samples_y)
                all_info.append(batch["info"])
            
            all_generation = torch.cat(all_generation)
            all_samples_x = torch.cat(all_samples_x)
            all_samples_y = torch.cat(all_samples_y)
            all_info = torch.cat(all_info)
            NACRPS, MSE = calc_metrics(is_test, all_generation, all_samples_y)
            if foldername:
                with open("preprocess/data/var.pkl", "rb") as file:
                    variable_names, target_ids = pickle.load(file)
                with open("preprocess/data/mean_std.pkl", "rb") as file:
                    means, stds = pickle.load(file)
                reference_path = "preprocess/data/evaluation_reference_scale.pkl"
                reference_scales = None
                reference_metadata = None
                try:
                    with open(reference_path, "rb") as file:
                        reference = pickle.load(file)
                    reference_scales = reference["scales"]
                    reference_metadata = {
                        key: reference.get(key)
                        for key in ("version", "seed", "source")
                    }
                except FileNotFoundError:
                    pass
                detailed = calculate_predictive_metrics(
                    all_generation, all_samples_y, variable_names, target_ids,
                    means, stds, reference_scales=reference_scales, info=all_info,
                )
                split_name = "test" if is_test == 1 else "validation"
                report = {
                    "split": split_name,
                    "NACRPS": NACRPS,
                    "NACRPS_statistics": legacy_nacrps_statistics(is_test, all_generation, all_samples_y),
                    "NACRPS_definition": (
                        "Legacy quantile score: mean over q=.05..95 step .05 on test "
                        "or q=.25,.50,.75 on validation; normalized by sum(abs(valid standardized targets))."
                    ),
                    "MSE_mean_standardized_legacy": float(MSE.item()),
                    "evaluation_reference_scale": reference_metadata,
                    "predictive_metrics": detailed,
                }
                with open(foldername + "/metrics_" + split_name + "_nsample" + str(nsample) + ".json", "w", encoding="utf-8") as file:
                    json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=True)
            if is_test == 1:
                pickle.dump([all_generation, all_samples_y, all_samples_x], open(foldername + "/generated_outputs" + str(nsample) + ".pkl", "wb"))
                pickle.dump([NACRPS, MSE], open(foldername + "/result_nsample" + str(nsample) + ".pkl", "wb"))
            return NACRPS, MSE
