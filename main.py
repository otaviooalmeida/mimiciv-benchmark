import argparse
import torch
import datetime
import json
import yaml
import os
from diff import TDSTF
from dataset import get_dataloader
from exe import train, evaluate
from reproducibility import seed_everything, validate_split_seed

parser = argparse.ArgumentParser(description='')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--modelfolder', type=str, default='')
parser.add_argument('--nsample', type=int, default=100)
parser.add_argument('--seed', type=int, default=None, help='Seed for training and test-time sampling.')
args = parser.parse_args()
config = yaml.safe_load(open('config/base.yaml', 'r'))
args.seed = config.get('seed', 2026) if args.seed is None else args.seed
if args.seed < 0:
    parser.error('--seed must be a non-negative integer')
seed_everything(args.seed)
print(args)
print(json.dumps(config, indent=4))
split_seed = validate_split_seed('preprocess/data/splits.pkl', args.seed)
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
foldername = './save/attention_' + current_time + '/'
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + '/run_metadata.json', 'w', encoding='utf-8') as metadata_file:
    json.dump({
        'seed': args.seed,
        'data_seed': args.seed,
        'split_seed': split_seed,
        'deterministic_algorithms': True,
        'config': config,
    }, metadata_file, indent=2)
data_path = 'preprocess/data/dataset.pkl'
var_path = 'preprocess/data/var.pkl'
size = config['diffusion']['size']
train_loader, valid_loader, test_loader = get_dataloader(
    data_path,
    var_path,
    size,
    batch_size=config['train']['batch_size'],
    seed=args.seed,
)
model = TDSTF(config, args.device).to(args.device)

if args.modelfolder == '':
    train(
        model,
        config,
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        seed=args.seed,
    )
else:
    model.load_state_dict(torch.load('./save/' + args.modelfolder + '/model.pth'))

print('test')
NACRPS, MSE = evaluate(
    1, model, test_loader, nsample=args.nsample, foldername=foldername, seed=args.seed
)
print('NACRPS (legado): {}'.format(NACRPS))
print('MSE da média, normalizado (legado): {}'.format(MSE))
