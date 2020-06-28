from optimizers import (KFACOptimizer, EKFACOptimizer)
import torch

def get_optimizer(optim_name, net, args):
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.learning_rate,
                                     betas=(args.momentum, 0.999),
                                     eps=1e-8,
                                     weight_decay=args.weight_decay,
                                     amsgrad=False)
    elif optim_name == 'kfac':
        optimizer = KFACOptimizer(net,
                                  lr=args.learning_rate,
                                  momentum=args.momentum,
                                  stat_decay=args.stat_decay,
                                  damping=args.damping,
                                  kl_clip=args.kl_clip,
                                  weight_decay=args.weight_decay,
                                  TCov=args.TCov,
                                  TInv=args.TInv)
    elif optim_name == 'ekfac':
        optimizer = EKFACOptimizer(net,
                                   lr=args.learning_rate,
                                   momentum=args.momentum,
                                   stat_decay=args.stat_decay,
                                   damping=args.damping,
                                   kl_clip=args.kl_clip,
                                   weight_decay=args.weight_decay,
                                   TCov=args.TCov,
                                   TScal=args.TScal,
                                   TInv=args.TInv)
    else:
        raise NotImplementedError
    return optimizer
