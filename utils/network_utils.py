from models.cifar import (alexnet, densenet, resnet, wrn,
                          vgg11, vgg11_bn, vgg13, vgg13_bn,
                          vgg16, vgg16_bn, vgg19_bn, vgg19,
                          simple_cnn)

def get_network(network, **kwargs):
    networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'wrn': wrn,
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19,
        'vgg11_bn': vgg11_bn,
        'vgg13_bn': vgg13_bn,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'simple_cnn': simple_cnn
    }

    return networks[network](**kwargs)

