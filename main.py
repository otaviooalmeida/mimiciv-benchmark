import argparse
import torch
import datetime
import json
import yaml
import os
import pickle
from diff import TDSTF
from dataset import get_dataloader
from exe import train, evaluate
from baselines import evaluate_baselines
from reproducibility import seed_everything, validate_split_seed

parser = argparse.ArgumentParser(description='')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--modelfolder', type=str, default='')
parser.add_argument('--nsample', type=int, default=100)
parser.add_argument('--seed', type=int, default=None, help='Seed for training and split-time sampling.')
parser.add_argument('--split', choices=('val_model', 'calibration', 'test'), default='val_model',
                    help='Evaluation population. Test is accessed only when explicitly selected.')
args = parser.parse_args()
config = yaml.safe_load(open('config/base.yaml', 'r'))
args.seed = config.get('seed', 2026) if args.seed is None else args.seed
if args.seed < 0:
    parser.error('--seed must be a non-negative integer')
seed_everything(args.seed)
print(args)
print(json.dumps(config, indent=4))
split_seed = validate_split_seed('preprocess/data/splits.pkl', args.seed)
if not os.path.isfile('preprocess/data/evaluation_reference_scale.pkl'):
    parser.error('Frozen evaluation scales are missing; rerun preprocess/step_4.py before training/evaluation')
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
foldername = './save/attention_' + current_time + '/'
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + '/run_metadata.json', 'w', encoding='utf-8') as metadata_file:
    json.dump({
        'seed': args.seed,
        'data_seed': args.seed,
        'split_seed': split_seed,
        'evaluation_split': args.split,
        'deterministic_algorithms': True,
        'config': config,
    }, metadata_file, indent=2)
data_path = 'preprocess/data/dataset.pkl'
var_path = 'preprocess/data/var.pkl'
size = config['diffusion']['size']
train_loader, val_model_loader, calibration_loader, test_loader = get_dataloader(
    data_path,
    var_path,
    size,
    batch_size=config['train']['batch_size'],
    seed=args.seed,
    include_test=args.split == 'test',
)
evaluation_loaders = {
    'val_model': val_model_loader,
    'calibration': calibration_loader,
    'test': test_loader,
}
evaluation_loader = evaluation_loaders[args.split]
if evaluation_loader is None:
    parser.error("split '{}' is unavailable; regenerate the dataset with preprocess/step_4.py".format(args.split))
model = TDSTF(config, args.device).to(args.device)

if args.modelfolder == '':
    train(
        model,
        config,
        train_loader,
        valid_loader=val_model_loader,
        foldername=foldername,
        seed=args.seed,
    )
else:
    model.load_state_dict(torch.load('./save/' + args.modelfolder + '/model.pth'))

print('evaluation split:', args.split)
NACRPS, MSE = evaluate(
    int(args.split == 'test'), model, evaluation_loader, nsample=args.nsample,
    foldername=foldername, seed=args.seed,
    tail_thresholds=config.get('evaluation', {}).get('tail_thresholds'),
    split_name=args.split,
)
print('NACRPS (legado): {}'.format(NACRPS))
print('MSE da média, normalizado (legado): {}'.format(MSE))

with open('preprocess/data/var.pkl', 'rb') as file:
    variable_names, target_ids = pickle.load(file)
with open('preprocess/data/mean_std.pkl', 'rb') as file:
    means, stds = pickle.load(file)
with open('preprocess/data/evaluation_reference_scale.pkl', 'rb') as file:
    reference_scales = pickle.load(file)['scales']
evaluate_baselines(
    train_loader, evaluation_loader, variable_names, target_ids, means, stds,
    reference_scales, args.split, os.path.join(foldername, 'baseline_metrics.json'),
    tail_thresholds=config.get('evaluation', {}).get('tail_thresholds'),
)
