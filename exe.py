import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import time

from reproducibility import seed_everything


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername='',
    seed=2026,
    abrupt_weighting=False,
):
    fit_start = time.perf_counter()
    seed_everything(seed)
    device = next(model.parameters()).device
    train_seconds = 0.0
    validation_seconds = 0.0
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
        epoch_start = time.perf_counter()
        synchronize(device)
        train_start = time.perf_counter()
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(
                    train_batch, config['diffusion']['size'], size_y,
                    abrupt_weighting=abrupt_weighting,
                )
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
        synchronize(device)
        epoch_train_seconds = time.perf_counter() - train_start
        train_seconds += epoch_train_seconds
        epoch_validation_seconds = 0.0
        stop_early = False
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            synchronize(device)
            validation_start = time.perf_counter()
            CRPS_valid, _ = evaluate(
                0, model, valid_loader, nsample=5, foldername=foldername, seed=seed + 1
            )
            synchronize(device)
            epoch_validation_seconds = time.perf_counter() - validation_start
            validation_seconds += epoch_validation_seconds
            print('{} (best)'.format(round(best_valid_loss, 4)))
            print('{} (current)'.format(round(CRPS_valid, 4)))
            if best_valid_loss > CRPS_valid:
                ct = 0
                best_valid_loss = CRPS_valid
                torch.save(model.state_dict(), output_path)
                print('model updated')
            else:
                ct += 1
                print('count: {}'.format(ct))
            # earlystopping
            if ct > 2:
                print('stop')
                stop_early = True
        print(
            'Timing epoch {}: train={:.1f}s, validation={:.1f}s, wall={:.1f}s'.format(
                epoch_no + 1, epoch_train_seconds, epoch_validation_seconds,
                time.perf_counter() - epoch_start,
            )
        )
        if stop_early:
            break

    if valid_loader is not None and best_valid_loss < np.inf:
        model.load_state_dict(torch.load(output_path))
    synchronize(device)
    print(
        'Fit timing: train={:.1f}s, validation={:.1f}s, total={:.1f}s'.format(
            train_seconds, validation_seconds, time.perf_counter() - fit_start,
        )
    )

def calc_validation_metrics(all_generation, all_samples_y, all_info):
    target = all_samples_y[:, 2]
    mask = all_samples_y[:, 3].bool()
    sorted_generation = all_generation.sort(dim=-1).values
    n_samples = all_generation.shape[-1]
    coefficients = 2 * torch.arange(1, n_samples + 1, device=all_generation.device) - n_samples - 1
    pairwise_term = (sorted_generation * coefficients).sum(dim=-1) / n_samples ** 2
    point_crps = (all_generation - target.unsqueeze(-1)).abs().mean(dim=-1) - pairwise_term
    abrupt = all_info[:, -1].to(all_generation.device).bool() if all_info.shape[1] >= 4 else torch.zeros(
        len(all_generation), dtype=torch.bool, device=all_generation.device
    )

    def mean_for(windows):
        selected = mask & windows.unsqueeze(1)
        return point_crps[selected].mean().item() if selected.any() else float('nan')

    lower = torch.quantile(all_generation, 0.025, dim=-1)
    upper = torch.quantile(all_generation, 0.975, dim=-1)
    coverage = (((target >= lower) & (target <= upper)) & mask).sum() / mask.sum()
    return {
        'overall_crps': mean_for(torch.ones_like(abrupt)),
        'abrupt_crps': mean_for(abrupt),
        'normal_crps': mean_for(~abrupt),
        'coverage_95': coverage.item(),
    }


def calc_metrics(is_test, all_generation, all_samples_y):
    MSE = None
    target = all_samples_y[:, 2]
    if is_test == 1:
        quantiles = np.arange(0.05, 1.0, 0.05)
        # calculate MSE
        gt = all_samples_y[:, 2]
        mask = all_samples_y[:, 3]
        prediction = all_generation.mean(dim=2)
        MSE = ((prediction - gt) * mask) ** 2
        MSE = MSE.sum() / mask.sum()
    else:
        quantiles = np.arange(0.25, 1.0, 0.25)
    denom = torch.sum(torch.abs(target))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(all_generation)):
            q_pred.append(torch.quantile(all_generation[j], quantiles[i], dim = -1))
        q_pred = torch.cat(q_pred, 0).reshape(-1)
        target = target.reshape(-1)
        q_loss = 2 * torch.sum(torch.abs((q_pred - target) * all_samples_y[:, 3].reshape(-1) * ((target <= q_pred) * 1.0 - quantiles[i])))
        CRPS += q_loss / denom
    
    return CRPS.item() / len(quantiles), MSE

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
                if not is_test:
                    all_info.append(batch['info'])
            
            all_generation = torch.cat(all_generation)
            all_samples_x = torch.cat(all_samples_x)
            all_samples_y = torch.cat(all_samples_y)
            if is_test:
                CRPS, MSE = calc_metrics(is_test, all_generation, all_samples_y)
            else:
                CRPS, MSE = calc_metrics(is_test, all_generation, all_samples_y)
                metrics = calc_validation_metrics(
                    all_generation, all_samples_y, torch.cat(all_info)
                )
                print(
                    'Validation ({} samples): overall CRPS={overall_crps:.4f}, '
                    'abrupt CRPS={abrupt_crps:.4f}, normal CRPS={normal_crps:.4f}, '
                    '95% coverage={coverage_95:.4f}'.format(nsample, **metrics)
                )
            if is_test == 1:
                pickle.dump([all_generation, all_samples_y, all_samples_x], open(foldername + "/generated_outputs" + str(nsample) + ".pkl", "wb"))
                pickle.dump([CRPS, MSE], open(foldername + "/result_nsample" + str(nsample) + ".pkl", "wb"))
            return CRPS, MSE
