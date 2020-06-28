from torch.optim.lr_scheduler import (MultiStepLR, ReduceLROnPlateau)

def get_lr_scheduler(optimizer, args):
    if args.lr_sched == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer,
                                         mode='min',
                                         factor=args.lr_sched_factor,
                                         patience=args.lr_sched_patience,
                                         min_lr=1e-7)
    elif args.lr_sched == 'multistep':
        if args.milestone is None:
            lr_scheduler = MultiStepLR(optimizer,
                                       milestones=[int(args.epoch*0.5), int(args.epoch*0.75)],
                                       gamma=0.1)
        else:
            milestone = [int(_) for _ in args.milestone.split(',')]
            lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)
    else:
        raise NotImplementedError
    return lr_scheduler
