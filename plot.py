from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

sns.set()
# sns.set_theme()

# for font in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
#     print(font)

# specify the custom font to use
# fontpath = '/usr/share/fonts/nerd-fonts-complete/TTF/Sauce Code Pro Nerd Font Complete.ttf'
# prop = font_manager.FontProperties(fname=fontpath)
# plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['text.usetex'] = True

# logdir = 'toy/simple_mlp128/'
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='toy/simple_mlp128/', type=str)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--stop', default=100, type=int)
parser.add_argument('--clip', default=None, type=int)
parser.add_argument('--title', default=None, type=str)
parser.add_argument('--logscale', default=True, type=bool)
args = parser.parse_args()
logdir = args.logdir
start = args.start
stop = args.stop
clip = args.clip
title = args.title
logscale = args.logscale

# create output directory
directory = f"{logdir.replace('/', '_')}figs_{start}_{stop}"
if clip is not None:
    directory += f"_{clip}"
try:
    os.mkdir(directory)
except OSError:
    print ("Creation of the directory %s failed" % directory)
else:
    print ("Successfully created the directory %s " % directory)

# iterate over all directories and load data
data = {}
for subdir, dirs, files in os.walk(logdir):
    for file in files:
        print(subdir)
        event_acc = EventAccumulator(subdir)
        event_acc.Reload()
        optimizer = subdir.split('/')[2]
        if "gkfac" in optimizer:
            optimizer = "K-FAC (two-level)"
        elif "kfac" in optimizer:
            optimizer = "K-FAC (one-level)"
        elif "sgd" in optimizer:
            optimizer = "SGD"
        elif "adam" in optimizer:
            optimizer = "ADAM"
        data[optimizer] = {}
        for scalar in event_acc.Tags()['scalars']:
            # print(scalar)
            time, epoch, value = zip(*event_acc.Scalars(scalar))
            data[optimizer][scalar] = (np.asarray(time), np.asarray(epoch), np.asarray(value))

scalar_list = list(data["SGD"])

optimizer_list = ["SGD", "ADAM", "K-FAC (one-level)", "K-FAC (two-level)"]

scalar_dict = {"train/lr": "Learning Rate Schedule",
               "train/loss": "Training Loss",
               "train/acc": "Training Accuracy",
               "test/loss": "Validation Loss",
               "test/acc": "Validation Accuracy"}

for scalar in scalar_list:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for optimizer in optimizer_list:
        x = data[optimizer][scalar][1][start:stop]
        y = data[optimizer][scalar][2][start:stop]
        if "loss" in scalar and clip is not None:
            for i in reversed(range(len(y))):
                if y[i] > clip:
                    y[i] = 2 * y[i+1]
                    print(f"Warning( {optimizer}, {scalar} ): Clipping {i}-th component of loss.")
        plt.plot(x, y, label=optimizer)
    if title is not None:
        plt.title(title)
    else:
        plt.title(f"{scalar_dict[scalar]} vs Epochs")
    plt.xlabel("Epochs")
    h = plt.ylabel(scalar_dict[scalar])
    # h.set_rotation(0)
    if "Accuracy" in scalar_dict[scalar]:
        plt.legend(loc="lower right")
    else:
        if logscale:
            ax.set_yscale('log')
        plt.legend(loc="upper right")

    # plt.show()
    # define the name of the directory to be created
    plt.savefig(f"{directory}/{scalar.replace('/','_')}.png", dpi=300)
