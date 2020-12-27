'''Train deep autoencoder on MNIST dataset with PyTorch.'''
import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.get_args import get_args
from utils.data_utils import get_dataloader
from utils.optim_utils import get_optimizer
from utils.lr_scheduler_utils import get_lr_scheduler
from utils.log_utils import get_log_dir
from models.mnist import deep_autoencoder

def visualize_results(epoch, input_imgs, recon_imgs):
    plt.figure(figsize=(9, 2))

    input_imgs = input_imgs.view(-1, 1, 28, 28).cpu().detach().numpy()
    recon_imgs = recon_imgs.view(-1, 1, 28, 28).cpu().detach().numpy()

    for i, item in enumerate(input_imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        plt.imshow(item[0])

    for i, item in enumerate(recon_imgs):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(item[0])

    plt.savefig(f"{visualization_dir}/epoch_{epoch}.png")
    plt.close()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    desc = ('[%s][LR=%s] Loss: %.3f' % (tag, optimizer.param_groups[0]['lr'], 0))

    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, _) in prog_bar:
        inputs = inputs.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, inputs) # TODO(nikolas): Check
        if optim_name in ['kfac', 'ekfac', 'gkfac'] and optimizer.steps % optimizer.TCov == 0:
            # compute true Fisher
            optimizer.acc_stats = True
            with torch.no_grad():
                sampled_outputs = torch.bernoulli(torch.sigmoid(outputs))
            loss_sample = criterion(outputs, sampled_outputs)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true Fisher.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        desc = ('[%s][LR=%s] Loss: %.3f' %
                (tag, optimizer.param_groups[0]['lr'], train_loss / (batch_idx + 1)))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss / (batch_idx + 1), epoch)

def test(epoch):
    net.eval()
    test_loss = 0

    desc = ('[%s][LR=%s] Loss: %.3f' % (tag, optimizer.param_groups[0]['lr'], 0))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, _) in prog_bar:
            inputs = inputs.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, inputs) # TODO(nikolas): Check

            test_loss += loss.item()
            desc = ('[%s][LR=%s] Loss: %.3f'
                    % (tag, optimizer.param_groups[0]['lr'], test_loss / (batch_idx + 1)))
            prog_bar.set_description(desc, refresh=True)

            if batch_idx == 0:
                visualize_results(epoch, inputs, outputs)

    # Save checkpoint.
    test_loss = test_loss / (batch_idx + 1)

    writer.add_scalar('test/loss', test_loss, epoch)

    return test_loss

#
# main script
#
# get command-line arguments
args = get_args()
args.network = 'deep_autoencoder'
args.dataset = 'mnist'

# set random seed for reproducibility
torch.manual_seed(args.seed)

# init model
net = deep_autoencoder()
net = net.to(args.device)

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
criterion = torch.nn.BCEWithLogitsLoss()

# init summary writter
log_dir = get_log_dir(optim_name, args)

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# create output directory for visualized results
visualization_dir = f"visuals/mnist/{optim_name}"
if not os.path.isdir(visualization_dir):
    os.makedirs(visualization_dir)

if __name__ == '__main__':
    # start training
    for epoch in range(args.epoch):
        train(epoch)
        test_loss = test(epoch)
        if args.lr_sched == 'plateau':
            lr_scheduler.step(test_loss)
        else:
            lr_scheduler.step()

