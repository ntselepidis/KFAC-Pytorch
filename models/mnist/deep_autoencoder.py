import torch
import torch.nn as nn

"""
Deep autoencoder network for MNIST.
"""

def mlp(sizes, activation, bias=True):
    """Basic multilayer perceptron architecture."""
    layers = []
    for i in range(len(sizes)-2):
        layers += [nn.Linear(sizes[i], sizes[i+1], bias=bias), activation]
    layers += [nn.Linear(sizes[-2], sizes[-1], bias=bias)]
    return nn.Sequential(*layers)

class deep_autoencoder(nn.Module):

    def __init__(self, encoder_sizes=None, decoder_sizes=None,
            activation=nn.ReLU(), bias=True, seed=None, **kwargs):

        super(deep_autoencoder, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if encoder_sizes is None:
            encoder_sizes = [28 * 28, 1000, 500, 250, 30]

        if decoder_sizes is None:
            decoder_sizes = [30, 250, 500, 1000, 28 * 28]

        assert encoder_sizes[0] == decoder_sizes[-1]

        # setup encoder network
        self.encoder = mlp(encoder_sizes, activation, bias=bias)

        # setup decoder network
        self.decoder = mlp(decoder_sizes, activation, bias=bias)

        # store type of activation
        self.activation_type = str(activation).replace('()', '').lower()

        # initialize weights
        self.apply(lambda m: self.init_weights(m))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight,
                    gain=nn.init.calculate_gain(self.activation_type))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

