import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Simple MLP for cifar dataset.
"""

class SimpleMLP(nn.Module):
    def __init__(self, d_in, d_out, d_h, n_h, bias=True, batch_norm=True, activation=nn.ReLU(), seed=None):
        super(SimpleMLP, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.layers = nn.ModuleList()
        if n_h > 0:
            self.layers.append(nn.Linear(d_in, d_h, bias=bias))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(num_features=d_h))
            if activation is not None:
                self.layers.append(activation)
            for i in range(n_h):
                self.layers.append(nn.Linear(d_h, d_h, bias=bias))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(num_features=d_h))
                if activation is not None:
                    self.layers.append(activation)
            self.layers.append(nn.Linear(d_h, d_out, bias=bias))
        else:
            self.layers.append(nn.Linear(d_in, d_out, bias=bias))
        self.apply(self.init_weights)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            # m.weight.data.uniform_(-stdv, stdv)
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)

def simple_mlp(num_classes, depth, hidden_dim, **kwargs):
    """
    Constructs a SimpleMLP model.
    """
    return SimpleMLP(d_in=3*32*32, d_out=num_classes, d_h=hidden_dim, n_h=depth, seed=0)

if __name__ == '__main__':
    bs = 4         # Batch size
    d_in = 3*32*32 # Dimension of input
    d_out = 10     # Dimension of output
    d_h = 5        # Dimension of each hidden layer
    n_h = 2        # Number of hidden layers

    dtype = torch.float
    device = torch.device("cpu")

    x = torch.ones(bs, d_in, dtype=dtype, device=device)

    # net = simple_mlp(d_in=d_in, d_out=d_out, d_h=d_h, n_h=n_h, seed=0)
    net = simple_mlp(num_classes=d_out, depth=n_h, hidden_dim=16*16)
    print(net)

    y = net(x)
    print(y)
