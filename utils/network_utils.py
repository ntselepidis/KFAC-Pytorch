from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn,
                          wrn, simple_cnn)


def get_network(network, **kwargs):
    networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn,
        'simple_cnn': simple_cnn

    }

    return networks[network](**kwargs)

