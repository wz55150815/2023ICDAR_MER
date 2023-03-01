import os
import time
import argparse
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from dataset import get_dataset
from models.Backbone import Backbone
from training import train, eval
from utils import load_config, save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser(description='HYB Tree')
parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
parser.add_argument('--check', default=True, help='only for code check')
args = parser.parse_args()

if not args.config:
    print('please provide config yaml')
    exit(-1)

"""config"""
params = load_config(args.config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

train_loader, eval_loader = get_dataset(params)
model = Backbone(params).to(device)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_' \
             f'max_size-{params["image_height"]}-{params["image_width"]}'
print(model.name)
print("model total parameters:", sum((x.numel() for x in model.parameters())))

if args.check:
    writer = None
else:
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']),
                                                      weight_decay=float(params['weight_decay']))

if params['finetune'] and Path(".//model.pkl").exists():
    print('loading pretrain model weight')
    model.load_state_dict(torch.load(".//model.pkl"))

min_score = 0
min_step = 0
for epoch in range(params['epochs']):
    train_loss, train_word_score, train_node_score, train_expRate = train(params, model, optimizer, epoch, train_loader, writer=writer)
    torch.save(model.state_dict(), ".//model.pkl")
    if epoch > 150:
        eval_loss, eval_word_score, eval_node_score, eval_expRate = eval(params, model, epoch, eval_loader,
                                                                         writer=writer)
        print(
            f'Epoch: {epoch + 1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  struct score: {eval_node_score:.4f} '
            f'ExpRate: {eval_expRate:.4f}')

        if eval_expRate >= min_score:
            min_score = eval_expRate
            min_step = 0

        elif min_score != 0 and 'lr_decay' in params and params['lr_decay'] == 'step':
            min_step += 1

            if min_step > params['step_ratio']:
                new_lr = optimizer.param_groups[0]['lr'] / params['step_decay']

                if new_lr < params['lr'] / 1000:
                    print('lr is too small')
                    exit(-1)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                min_step = 0
