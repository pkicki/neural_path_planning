import inspect
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.planner import PlanningNetworkMP
from utils.execution import ExperimentHandler
import numpy as np
from matplotlib import pyplot as plt
from utils.test import read_map, _plot, run_and_plot
import tensorflow as tf

tf.random.set_seed(444)


if __name__ == '__main__':
    p = "../data/exemplary_paths/x/"
    model_path = "./trained_models/corl_N_6/best-28"
    c = range(14)
    map_path = [p + str(x) + ".png" for x in c]
    fig, axes = plt.subplots(nrows=2, ncols=7, gridspec_kw={"hspace": 0.015, "wspace": 0.015})
    #     1    2    3    4     5   6    7    8    9    10   11   12   13   14
    xs = [13.5, 19., 14., 16., 19., 19., 16., 15., 14., 18., 5., 19.5, 16., 19.]
    ys = [9.,  5.5, 9.5,  -2., 2.,  -5., 3.,  9., -3.5,  -1., 1., 0.5,  -3.,  0.]
    ths = [np.pi/2, 0., np.pi/4., -np.pi/6, 0., -np.pi/10., -np.pi/5, np.pi/3., -np.pi/20., np.pi/10, np.pi/6., np.pi/4., 0., -np.pi/6]
    #         1     2      3         4      5        6      7            8         9           10     11           12     13     14
    for i, ax in enumerate(axes.flat):
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft = False)
        im = run_and_plot(model_path, map_path[i], None, xs[i], ys[i], ths[i], ax)
    plt.show()
