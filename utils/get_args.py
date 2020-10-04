import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', default='vgg16_bn', type=str)
    parser.add_argument('--depth', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str)

    # densenet
    parser.add_argument('--growthRate', default=12, type=int)
    parser.add_argument('--compressionRate', default=2, type=int)

    # wrn, densenet
    parser.add_argument('--widen_factor', default=1, type=int)
    parser.add_argument('--dropRate', default=0.0, type=float)

    # simple_mlp
    parser.add_argument('--hidden_dim', default=16*16, type=int)
    parser.add_argument('--activation', default=None, type=str,
            choices=['relu', 'sigmoid', 'tanh'])

    # General utils
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--log_dir', default='runs/pretrain', type=str)

    # Algorithms and hyperparameters
    parser.add_argument('--optimizer', default='kfac', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=100, type=int)

    # Learning rate scheduler (ReduceLROnPlateau, MultiStepLR)
    parser.add_argument('--lr_sched', default='plateau', type=str,
            choices=['plateau', 'multistep'])

    # ReduceLROnPlateau params
    parser.add_argument('--lr_sched_factor', default=0.1, type=float)
    parser.add_argument('--lr_sched_patience', default=10, type=int)
    # MultiStepLR params
    parser.add_argument('--lr_sched_milestone', default=None, type=str)

    # Hyperparameters
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--stat_decay', default=0.95, type=float)
    parser.add_argument('--damping', default=1e-3, type=float)
    parser.add_argument('--kl_clip', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=3e-3, type=float)
    parser.add_argument('--TCov', default=10, type=int)
    parser.add_argument('--TScal', default=10, type=int)
    parser.add_argument('--TInv', default=100, type=int)

    # Weights for combining fine and coarse solutions in GKFAC
    # GKFAC solution = omega_1 * fine + omega_2 * coarse
    parser.add_argument('--omega_1', default=1.0, type=float)
    parser.add_argument('--omega_2', default=1.0, type=float)

    # Technique for inverting diagonal blocks of KFAC
    # solver = 'symeig' or 'approx'
    parser.add_argument('--solver', default='symeig', type=str, choices=['symeig', 'approx'])

    # Algorithm for downsampling tensors in GKFAC
    # mode = 'nearest' or 'area'
    parser.add_argument('--mode', default='nearest', type=str, choices=['nearest', 'area'])

    parser.add_argument('--prefix', default=None, type=str)

    return parser.parse_args()
