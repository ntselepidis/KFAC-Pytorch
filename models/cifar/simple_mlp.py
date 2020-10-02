import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Simple MLP.
"""

class SimpleMLP(nn.Module):
    def __init__(self, d_in, d_out, d_h, n_h, bias=True, activation=nn.ReLU(), seed=None):
        super(SimpleMLP, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.activation = activation
        self.layers = nn.ModuleList()
        if n_h > 0:
            self.layers.append(nn.Linear(d_in, d_h, bias=bias))
            for i in range(n_h):
                self.layers.append(nn.Linear(d_h, d_h, bias=bias))
            self.layers.append(nn.Linear(d_h, d_out, bias=bias))
        else:
            self.layers.append(nn.Linear(d_in, d_out, bias=bias))

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        if self.activation is not None:
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
        else:
            for layer in self.layers[:-1]:
                x = layer(x)
        return self.layers[-1](x)

def simple_mlp(num_classes, depth, hidden_dim, **kwargs):
    """
    Constructs a SimpleMLP model.
    """
    return SimpleMLP(d_in=3*32*32, d_out=num_classes, d_h=hidden_dim, n_h=depth)

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
