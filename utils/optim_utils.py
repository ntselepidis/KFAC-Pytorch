from optimizers import (KFAC, EKFAC, GKFAC)
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
        optimizer = KFAC(net,
                         lr=args.learning_rate,
                         momentum=args.momentum,
                         stat_decay=args.stat_decay,
                         damping=args.damping,
                         kl_clip=args.kl_clip,
                         weight_decay=args.weight_decay,
                         TCov=args.TCov,
                         TInv=args.TInv,
                         solver=args.solver)
    elif optim_name == 'gkfac':
        optimizer = GKFAC(net,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          stat_decay=args.stat_decay,
                          damping=args.damping,
                          kl_clip=args.kl_clip,
                          weight_decay=args.weight_decay,
                          TCov=args.TCov,
                          TInv=args.TInv,
                          solver=args.solver,
                          omega_1=args.omega_1,
                          omega_2=args.omega_2,
                          mode=args.mode,
                          device=args.device)
    elif optim_name == 'ekfac':
        optimizer = EKFAC(net,
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
