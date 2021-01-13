'''Train a neural network on CIFAR10 or CIFAR100 with PyTorch.'''
import os
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.get_args import get_args
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from utils.optim_utils import get_optimizer
from utils.lr_scheduler_utils import get_lr_scheduler
from utils.log_utils import get_log_dir


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
            # compute true Fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1), 1).squeeze()
            loss_sample = criterion(outputs, sampled_y)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true Fisher.
        loss.backward()
        optimizer.step()

        train_loss += loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, optimizer.param_groups[0]['lr'], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag, optimizer.param_groups[0]['lr'], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, optimizer.param_groups[0]['lr'], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    test_loss = test_loss / (batch_idx + 1)
    acc = 100. * correct / total

    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/acc', acc, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': test_loss,
            'args': args
        }

        torch.save(state, '%s/%s_%s_%s%s_best.t7' % (log_dir,
                                                     args.optimizer,
                                                     args.dataset,
                                                     args.network,
                                                     args.depth))
        best_acc = acc

    return test_loss

#
# main script
#
# get command-line arguments
args = get_args()

# set random seed for reproducibility
torch.manual_seed(args.seed)

# init model
num_classes = { 'cifar10': 10, 'cifar100': 100 }

net = get_network(
        args.network,
        depth=args.depth,
        num_classes=num_classes[args.dataset],
        growthRate=args.growthRate,
        compressionRate=args.compressionRate,
        widen_factor=args.widen_factor,
        dropRate=args.dropRate,
        hidden_dim=args.hidden_dim
    ).to(args.device)

# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)
# init optimizer
optim_name = args.optimizer.lower()
tag = optim_name
optimizer = get_optimizer(optim_name, net, args)

# init lr scheduler
lr_scheduler = get_lr_scheduler(optimizer, args)

# init criterion
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

# init summary writter
log_dir = get_log_dir(optim_name, args)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# start training
for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test_loss = test(epoch)
    if args.lr_sched == 'plateau':
        lr_scheduler.step(test_loss)
    else:
        lr_scheduler.step()
