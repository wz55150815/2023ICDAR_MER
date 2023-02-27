import os
import tempfile
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
from distributed_utils import init_distributed_mode, dist, cleanup

parser = argparse.ArgumentParser(description='HYB Tree')
parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
parser.add_argument('--check', default=True, help='only for code check')
# 是否启用SyncBatchNorm,启用的话速度会变慢，但是可以替高正确率
parser.add_argument('--syncBN', type=bool, default=True)
# 不要改该参数，系统会自动分配
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
# 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
parser.add_argument('--world-size', default=4, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
args = parser.parse_args()

if not args.config:
    print('please provide config yaml')
    exit(-1)

"""config"""
# 初始化环境
init_distributed_mode(args=args)

rank = args.rank

params = load_config(args.config)

device = torch.device(args.device)
params['device'] = device
# 这里需要改
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

# 载入预训练权重
if params['finetune'] and Path(".//model.pkl").exists():
    print('loading pretrain model weight')
    model.load_state_dict(torch.load(".//model.pkl"))
else:
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

if args.syncBN:
    # 使用SyncBatchNorm后训练会更耗时
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
# 包装模型，转为DDP模型
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']),
                                                      weight_decay=float(params['weight_decay']))
min_score = 0
min_step = 0
for epoch in range(params['epochs']):
    # 需要增加train_sampler

    train_loss, train_word_score, train_node_score, train_expRate = train(params, model, optimizer, epoch, train_loader,
                                                                          writer=writer)
    if rank == 0:
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

# 删除临时缓存文件
if rank == 0:
    if os.path.exists(checkpoint_path) is True:
        os.remove(checkpoint_path)
# 释放进程
cleanup()
