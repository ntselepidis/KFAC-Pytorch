import argparse
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--network', type=str, default='vgg16_bn')
parser.add_argument('--optimizer', type=str, default='kfac')
parser.add_argument('--machine', type=str, default='dalabgpu')

args = parser.parse_args()

# Flags
simple_cnn = ''
vgg11 = ''
vgg13 = ''
vgg16 = ''
vgg19 = ''
vgg11_bn = ''
vgg13_bn = ''
vgg16_bn = ''
vgg19_bn = ''
resnet8 = '--depth 8'
resnet110 = '--depth 110'
wrn = '--depth 28 --widen_factor 10 --dropRate 0.3'
densenet = '--depth 100 --growthRate 12'

flag_dict = {
    'simple_cnn': simple_cnn,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    'resnet8': resnet8,
    'resnet110': resnet110,
    'wrn': wrn,
    'densenet': densenet
}

def grid_search(args):
    runs = []
    # Parameters
    batch_sizes = [32, 128, 512]
    momentums = [0.0, 0.9]
    learning_rates = [0.001, 0.01]
    wd = 1e-4
    TInvs = [100]
    flags = flag_dict[args.network]

    # Temporary hack
    args_network = args.network
    if 'resnet' in args_network:
        args_network = 'resnet'

    if args.optimizer in ['kfac', 'ekfac']:
        template = 'python main.py ' \
                   '--dataset %s ' \
                   '--network %s ' \
                   '--optimizer %s ' \
                   '--batch_size %d ' \
                   '--epoch 100 ' \
                   '--learning_rate %f ' \
                   '--momentum %f ' \
                   '--weight_decay %f ' \
                   '--TInv %d %s'
#                   '--milestone 40,80 '

        for bs in batch_sizes:
            for tinv in TInvs:
                for mom in momentums:
                    for lr in learning_rates:
                        runs.append(template % (args.dataset, args_network, args.optimizer, bs, lr, mom, wd, tinv, flags))

    elif args.optimizer in ['sgd', 'adam']:
        template = 'python main.py ' \
                   '--dataset %s ' \
                   '--network %s ' \
                   '--optimizer %s ' \
                   '--batch_size %d ' \
                   '--epoch 100 ' \
                   '--learning_rate %f ' \
                   '--momentum %f ' \
                   '--weight_decay %f %s'
#                   '--milestone 60,120,180 '

        for bs in batch_sizes:
            for mom in momentums:
                for lr in learning_rates:
                    runs.append(template % (args.dataset, args_network, args.optimizer, bs, lr, mom, wd, flags))

    return runs


def gen_script(args, runs):
    with open('submit_%s_%s_%s.sh' % (args.dataset, args.network, args.optimizer), 'w') as f:
        if args.machine == 'dalabgpu':
            f.write('#/bin/bash\n')
        for cnt, run in enumerate(runs):
            if args.machine == 'dalabgpu':
                f.write('%s\n' % run)
            else:
                f.write('bsub -W 04:00 -n 2 -M 16GB -R "rusage[mem=8192,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -oo $SCRATCH/KFAC_jobs/%s_%s_%s_%d.txt %s\n'
                        % (args.dataset, args.network, args.optimizer, int(time.time()+cnt), run))


if __name__ == '__main__':
    gen_script(args, grid_search(args))

