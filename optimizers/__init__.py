from .kfac import KFACOptimizer
from .ekfac import EKFACOptimizer
from .gkfac import GKFACOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    elif name == 'gkfac':
        return GKFACOptimizer
    else:
        raise NotImplementedError
