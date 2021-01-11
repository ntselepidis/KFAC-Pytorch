from .kfac import KFAC
from .ekfac import EKFAC
from .gkfac import GKFAC


def get_optimizer(name):
    if name == 'kfac':
        return KFAC
    elif name == 'ekfac':
        return EKFAC
    elif name == 'gkfac':
        return GKFAC
    else:
        raise NotImplementedError
