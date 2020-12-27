import torch

"""
Deep autoencoder network for MNIST.
"""

class deep_autoencoder(torch.nn.Module):

    def __init__(self, seed=None, **kwargs):
        super(deep_autoencoder, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        # setup encoder network
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 1000, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1000, 500, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(500, 250, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(250, 30, bias=True)
        )

        # setup decoder network
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(30, 250, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(250, 500, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(500, 1000, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1000, 28 * 28, bias=True)
        )

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight,
                    gain=torch.nn.init.calculate_gain('sigmoid'))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

