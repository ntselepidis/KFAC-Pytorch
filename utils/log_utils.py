import os

def get_log_dir(optim_name, args):
    # Network depth
    args_depth = "" if args.depth == 0 else str(args.depth)
    # Logging based on chosen optimizer
    if optim_name == 'sgd' or optim_name == 'adam':
        log_dir = os.path.join(args.log_dir, args.dataset, args.network + args_depth, args.optimizer,
                               'bs%d_lr%.4f_mom%.2f_wd%.4f' %
                               (args.batch_size,
                                args.learning_rate,
                                args.momentum,
                                args.weight_decay))
    elif optim_name == 'kfac':
        log_dir = os.path.join(args.log_dir, args.dataset, args.network + args_depth, args.optimizer,
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
        log_dir = os.path.join(args.log_dir, args.dataset, args.network + args_depth, args.optimizer,
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
