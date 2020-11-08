from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import os
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default='toy/simple_mlp128/', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--stop', default=100, type=int)
    parser.add_argument('--clip', default=None, type=int)
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--logscale', default=True, type=bool)
    args = parser.parse_args()
    return args

def create_output_directory(args):
    directory = f"{args.logdir.replace('/', '_')}_figs_{args.start}_{args.stop}"
    if args.clip is not None:
        directory += f"_{args.clip}"
    try:
        os.mkdir(directory)
    except OSError:
        print ("Creation of the directory %s failed" % directory)
    else:
        print ("Successfully created the directory %s " % directory)
    return directory

def get_optimizer(optimizer):
    if "gkfac" in optimizer:
        optimizer = "K-FAC (two-level)"
    elif "kfac" in optimizer:
        optimizer = "K-FAC (one-level)"
    elif "sgd" in optimizer:
        optimizer = "SGD"
    elif "adam" in optimizer:
        optimizer = "ADAM"
    else:
        pass
    return optimizer

def load_data(args):
    run = {"SGD": 0, "ADAM": 0, "K-FAC (one-level)": 0, "K-FAC (two-level)": 0}
    ind = {"SGD": 0, "ADAM": 1, "K-FAC (one-level)": 2, "K-FAC (two-level)": 3}
    df_list = [[] for _ in range(len(run))]
    for root, _, files in os.walk(args.logdir):
        for file in files:
            event_acc = EventAccumulator(root)
            event_acc.Reload()
            optimizer = get_optimizer(root)
            print(optimizer)
            print(root)
            data = {}
            data['optimizer'] = optimizer
            data['run'] = run[optimizer]
            run[optimizer] += 1
            for (i, scalar) in enumerate(event_acc.Tags()['scalars']):
                if scalar == 'train/lr':
                    continue
                times, epochs, values = zip(*event_acc.Scalars(scalar))
                if (i == 1):
                    data['times'] = np.asarray(times)[:args.stop]
                    data['epochs'] = np.asarray(epochs)[:args.stop]
                data[scalar] = np.asarray(values)[:args.stop]
            if run[optimizer] == 1:
                df = pd.DataFrame(data)
                df_list[ind[optimizer]] = df
            else:
                df = pd.DataFrame(data)
                df_list[ind[optimizer]] = pd.concat([df_list[ind[optimizer]], df])
    df = pd.concat(df_list)
    return df

def main():    
    # get arguments
    args = get_args()

    # iterate over all directories and load data
    df = load_data(args)

    df = df[(df['epochs'] >= args.start) & (df['epochs'] < args.stop)]

    print('Data Loaded')

    # utilities
    optimizer_list = ["SGD", "ADAM", "K-FAC (one-level)", "K-FAC (two-level)"]

    scalar_dict = {"train/loss": "Training Loss",
                   "train/acc": "Training Accuracy",
                   "test/loss": "Test Loss",
                   "test/acc": "Test Accuracy"}
 
    # create output directory
    directory = create_output_directory(args)
    
    # configure plots
    sns.set_style("darkgrid") 
    plt.rcParams['text.usetex'] = True

    for scalar in scalar_dict.keys():
        print(f'Generating plot for {scalar} ...')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.lineplot(data=df, x='epochs', y=scalar, hue='optimizer')
        if args.title is not None:
            plt.title(f"{args.title}: {scalar_dict[scalar]}", fontsize=17)
        else:
            plt.title(f"{scalar_dict[scalar]}", fontsize=17)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("")
        # h = plt.ylabel(scalar_dict[scalar])
        # h.set_rotation(0)
        if "Accuracy" in scalar_dict[scalar]:
            plt.legend(loc="lower right", fontsize=15)
        else:
            if args.logscale:
                ax.set_yscale('log')
            plt.legend(loc="upper right", fontsize=15)
        ax.xaxis.set_tick_params(labelsize=12.5)
        ax.yaxis.set_tick_params(labelsize=12.5)

        # plt.show()
        # save figure
        plt.savefig(f"{directory}/{scalar.replace('/','_')}.pdf")

if __name__ == '__main__':
    main()
