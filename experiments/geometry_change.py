import inspect
import os
import sys

from utils.test import run_and_plot

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
import tensorflow as tf
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

tf.random.set_seed(444)

matplotlib.rcParams.update({'font.size': 16})

if __name__ == '__main__':
    p = "../data/fig4/x/grid"
    model_path = "./trained_models/corl_N_6/best-28"
    c = [1, 2, 3, 4]
    map_path = [p + str(x) + ".png" for x in c]
    as_path = ["../data/fig4/as/as_left_test_" + str(x) + ".tsv" for x in c]
    fig, axes = plt.subplots(nrows=1, ncols=4, gridspec_kw={"hspace": 0.015, "wspace": 0.015})
    xs = [16.5, 20.5, 20.5, 20.5]
    ys = [9., 1.2, 1.2, 1.2]
    ths = [np.pi / 2, 0., 0., 0.]
    for i, ax in enumerate(axes.flat):
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft = False)
        im = run_and_plot(model_path, map_path[i], as_path[i], xs[i], ys[i], ths[i], ax)
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical")
    plt.show()
