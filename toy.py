'''Train toy dataset with PyTorch.'''
import argparse
import os
import sys
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.optim_utils import get_optimizer
from utils.lr_scheduler_utils import get_lr_scheduler
from utils.log_utils import get_log_dir
from models.cifar import SimpleMLP

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, nsamples, d_in, d_out, device):
        super().__init__()

        self.nsamples = nsamples

        X = torch.randn(nsamples, d_in, device=device)

        net = SimpleMLP(d_in, d_out, d_h=d_in, n_h=0, bias=False,
                batch_norm=False, activation=None, seed=1)

        net = net.to(device)

        self.X = X
        self.Y = net(self.X)

        # Considering classification instead of regression
        self.Y = torch.argmax(torch.nn.functional.softmax(self.Y, dim=1), axis=1)

        print('Ground Truth Model')
        print(net)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_args():
    # fetch args
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', default='simple_mlp', type=str)
    parser.add_argument('--dataset', default='toy', type=str)
    # simple_mlp
    parser.add_argument('--depth', default=0, type=int)
    parser.add_argument('--hidden_dim', default=16*16, type=int)

    # General utils
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--log_dir', default='runs/pretrain', type=str)

    # Algorithms and hyperparameters
    parser.add_argument('--optimizer', default='kfac', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=100, type=int)

    # Learning rate scheduler (ReduceLROnPlateau, MultiStepLR)
    parser.add_argument('--lr_sched', default='multistep', type=str, choices=['multistep'])
    # ReduceLROnPlateau params
    parser.add_argument('--lr_sched_factor', default=0.1, type=float)
    parser.add_argument('--lr_sched_patience', default=10, type=int)
    # MultiStepLR params
    parser.add_argument('--lr_sched_milestone', default=None, type=str)

    # Hyperparameters
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--stat_decay', default=0.95, type=float)
    parser.add_argument('--damping', default=1e-3, type=float)
    parser.add_argument('--kl_clip', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=3e-3, type=float)
    parser.add_argument('--TCov', default=10, type=int)
    parser.add_argument('--TScal', default=10, type=int)
    parser.add_argument('--TInv', default=100, type=int)

    # Weights for combining fine and coarse solutions in GKFAC
    # GKFAC solution = omega_1 * fine + omega_2 * coarse
    parser.add_argument('--omega_1', default=1.0, type=float)
    parser.add_argument('--omega_2', default=1.0, type=float)

    # Technique for inverting diagonal blocks of KFAC
    # solver = 'symeig' or 'approx'
    parser.add_argument('--solver', default='symeig', type=str, choices=['symeig', 'approx'])

    # Algorithm for downsampling tensors in GKFAC
    # mode = 'nearest' or 'area'
    parser.add_argument('--mode', default='nearest', type=str, choices=['nearest', 'area'])

    args = parser.parse_args()

    return args

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, optimizer.param_groups[0]['lr'], 0, 0, correct, total))

    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if optim_name in ['kfac', 'ekfac', 'gkfac'] and optimizer.steps % optimizer.TCov == 0:
            # compute true fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1), 1).squeeze()
            loss_sample = criterion(outputs, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true-fisher.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, optimizer.param_groups[0]['lr'], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)

# main script

# set random seed for reproducibility
torch.manual_seed(0)

# get command-line arguments
args = get_args()

# set main parameters
n_samples = 2500
d_in = 10 # 3 # Features
d_out = 2 # 1 # Classes

# init dataset
dataset = ToyDataset(n_samples, d_in, d_out, args.device)

# init data loader
trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# init model
# d_h = 20
n_h = args.depth

net = SimpleMLP(d_in, d_out, d_h=d_in, n_h=n_h, bias=False,
        batch_norm=True, activation=None, seed=0)

net = net.to(args.device)

print('Approximate Model')
print(net)

# init optimizer
optim_name = args.optimizer.lower()
tag = optim_name
optimizer = get_optimizer(optim_name, net, args)

# init lr scheduler
lr_scheduler = get_lr_scheduler(optimizer, args)

# init criterion
# criterion = torch.nn.MSELoss() # Regression
criterion = torch.nn.CrossEntropyLoss() # Classification

# init summary writter
log_dir = get_log_dir(optim_name, args)

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# start training
for epoch in range(args.epoch):
    train(epoch)
    lr_scheduler.step()

