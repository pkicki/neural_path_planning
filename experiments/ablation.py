import inspect
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.test import run_and_plot
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.random.set_seed(444)

if __name__ == '__main__':
    p = "../data/ablation/"
    model_path = "./trained_9d/best-71"
    c = [1, 2, 3, 4, 5]
    map_path = [p + str(x) + ".png" for x in c]
    fig, axes = plt.subplots(nrows=1, ncols=5, gridspec_kw={"hspace": 0.015, "wspace": 0.015})
    xs = [15.5, 20.5, 20.5, 20.5, 13.5]
    ys = [9., 1.2, 7.2, 3.2, -1.2]
    ths = [np.pi / 2, 0., 0., -np.pi / 4, np.pi / 4]
    for i, ax in enumerate(axes.flat):
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)
        im = run_and_plot(model_path, map_path[i], None, xs[i], ys[i], ths[i], ax)
    plt.show()
