'''Train toy dataset with PyTorch.'''
import os
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.get_args import get_args
from utils.optim_utils import get_optimizer
from utils.lr_scheduler_utils import get_lr_scheduler
from utils.log_utils import get_log_dir
from models.cifar import SimpleMLP

# Wraps custom dataset inside torch class for use in dataloader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Generates data from a ground truth model
def generate_data(n_samples, d_in, d_out):
    # init ground truth model
    net = SimpleMLP(d_in, d_out, d_h=d_in, n_h=0, bias=False,
            batch_norm=False, activation=None, seed=1)

    net = net.to(args.device)

    # generate dataset using ground truth model
    X = torch.randn(n_samples, d_in, device=args.device)
    Y = net(X)

    # generate target labels for classification
    if d_out == 1:
        Y = torch.as_tensor(torch.sigmoid(Y) > 0.5, dtype=torch.float32)#.squeeze()
    else:
        Y = torch.argmax(torch.nn.functional.softmax(Y, dim=1), axis=1)

    return X, Y

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
                # Perform one random experiment per output sample (row) in the batch, i.e.
                # draw one sample (without replacement) from the multinomial distribution,
                # given the row-wise softmax(output) as weights.
                # These weights represent the probability of each outcome.
                # Note: ``The multinomial distribution models the probability of counts
                # for each side of a k-sided dice rolled n times.``
                # In our case, k = outputs.shape[1] = d_out and n = 1.
                # torch.multinomial() returns the index of each side, which in our case
                # is the class target label of the sample.
                if d_out == 1:
                    sampled_targets = torch.bernoulli(torch.sigmoid(outputs))
                else:
                    sampled_targets = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1), 1).squeeze()
            loss_sample = criterion(outputs, sampled_targets)
            loss_sample.backward(retain_graph=True)
            optimizer.acc_stats = False
            optimizer.zero_grad()  # clear the gradient for computing true Fisher.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if d_out == 1:
            predicted = torch.as_tensor(torch.sigmoid(outputs) > 0.5, dtype=torch.float32)
        else:
            _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, optimizer.param_groups[0]['lr'], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)

def test(epoch):
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

            test_loss += loss.item()
            if d_out == 1:
                predicted = torch.as_tensor(torch.sigmoid(outputs) > 0.5, dtype=torch.float32)
            else:
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, optimizer.param_groups[0]['lr'], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    test_loss = test_loss / (batch_idx + 1)
    acc = 100. * correct / total

    writer.add_scalar('test/loss', test_loss, epoch)
    writer.add_scalar('test/acc', acc, epoch)

    return test_loss

#
# main script
#
# set random seed for reproducibility
torch.manual_seed(0)

# get command-line arguments
args = get_args()
args.network = 'simple_mlp'
args.dataset = 'toy'

# set main parameters
n_train = 25000
n_test = n_train // 10
n_samples = n_train + n_test
d_in = 10 # Features
d_out = 1 # Classes (for binary classification d_out can be either 1 or 2)

# generate data from ground truth model
X, Y = generate_data(n_samples, d_in, d_out)

# init trainset and testset
trainset = CustomDataset(X[:n_train], Y[:n_train])
testset  = CustomDataset(X[n_train:], Y[n_train:])

# init data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# choose activation fuction for approximate model
act_dict = {
        'relu': torch.nn.ReLU(),
        'sigmoid': torch.nn.Sigmoid(),
        'tanh': torch.nn.Tanh()
}

act = None if args.activation is None else act_dict[args.activation]

# init approximate model
net = SimpleMLP(d_in, d_out, d_h=d_in, n_h=args.depth, bias=False,
        batch_norm=True, activation=act, seed=0)

net = net.to(args.device)

# init optimizer
optim_name = args.optimizer.lower()
tag = optim_name
optimizer = get_optimizer(optim_name, net, args)

# init lr scheduler
lr_scheduler = get_lr_scheduler(optimizer, args)

# init criterion
# criterion = torch.nn.MSELoss() # Regression
if d_out == 1:
    criterion = torch.nn.BCEWithLogitsLoss() # Binary classification
else:
    criterion = torch.nn.CrossEntropyLoss() # Multi-class classification

# init summary writter
log_dir = get_log_dir(optim_name, args)

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

if __name__ == '__main__':
    # start training
    for epoch in range(args.epoch):
        train(epoch)
        test_loss = test(epoch)
        if args.lr_sched == 'plateau':
            lr_scheduler.step(test_loss)
        else:
            lr_scheduler.step()

