import os

def get_log_dir(optim_name, args):
    # Network depth
    args_depth = "" if args.depth == 0 else str(args.depth)
    # LR scheduler
    if args.lr_sched == 'plateau':
        lr_sched_str = 'ReduceLROnPlateau(%.1f,%d)' % (args.lr_sched_factor, args.lr_sched_patience)
    elif args.lr_sched == 'multistep':
        if args.lr_sched_milestone is None:
            milestone = str([int(args.epoch*0.5), int(args.epoch*0.75)]).replace(" ", "")
        else:
            milestone = '[' + args.lr_sched_milestone + ']'
        lr_sched_str = 'MultiStepLR' + milestone.replace('[', '(').replace(']', ')')
    else:
        raise NotImplementedError
    # Logging based on chosen optimizer
    if optim_name == 'sgd' or optim_name == 'adam':
        log_dir = os.path.join(args.log_dir, args.dataset, args.network + args_depth, args.optimizer, lr_sched_str,
                               'bs%d_lr%.4f_mom%.2f_wd%.4f' %
                               (args.batch_size,
                                args.learning_rate,
                                args.momentum,
                                args.weight_decay))
    elif optim_name == 'kfac' or optim_name == 'gkfac':
        args_optimizer = args.optimizer
        if optim_name == 'gkfac':
            args_optimizer = args_optimizer + '(' + str(args.omega) + ')'
        log_dir = os.path.join(args.log_dir, args.dataset, args.network + args_depth, args_optimizer, lr_sched_str,
                               'bs%d_lr%.4f_mom%.2f_wd%.4f_sd%.4f_dmp%.4f_kl%.4f_TCov%d_TInv%d' %
                               (args.batch_size,
                                args.learning_rate,
                                args.momentum,
                                args.weight_decay,
                                args.stat_decay,
                                args.damping,
                                args.kl_clip,
                                args.TCov,
                                args.TInv))
    elif optim_name == 'ekfac':
        log_dir = os.path.join(args.log_dir, args.dataset, args.network + args_depth, args.optimizer, lr_sched_str,
                               'bs%d_lr%.4f_mom%.2f_wd%.4f_sd%.4f_dmp%.4f_kl%.4f_TCov%d_TInv%d_TScal%d' %
                               (args.batch_size,
                                args.learning_rate,
                                args.momentum,
                                args.weight_decay,
                                args.stat_decay,
                                args.damping,
                                args.kl_clip,
                                args.TCov,
                                args.TInv,
                                args.TScal))
    return log_dir
