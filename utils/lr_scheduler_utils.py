from torch.optim.lr_scheduler import (MultiStepLR, ReduceLROnPlateau)

def get_lr_scheduler(optimizer, args):
    if args.milestone is None:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, min_lr=1e-7)
        #lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
    else:
        milestone = [int(_) for _ in args.milestone.split(',')]
        lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)
    return lr_scheduler
