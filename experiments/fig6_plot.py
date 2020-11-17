import inspect
import os
import sys
from random import shuffle
from time import time

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from utils.crucial_points import calculate_car_crucial_points

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import plan_loss, _plot, PlanningNetworkMP

from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from utils.execution import ExperimentHandler, LoadFromFile

tf.random.set_seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, list(data))
            pbar.update(batch_size)


#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def read_map(map_path):
    img = tf.io.read_file(map_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    free = img > 0.5
    obs = img < 0.5
    img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
    return img

def transform_to_img(x, y):
    x = 120 - (x / .2)
    y = 64 - (y / .2)
    return y, x

def _plot_car(cp, ax, c='m'):
    x = [p[0].numpy() for p in cp[1:]]
    y = [p[1].numpy() for p in cp[1:]]
    tmp = x[2]
    x[2] = x[3]
    x[3] = tmp
    tmp = y[2]
    y[2] = y[3]
    y[3] = tmp
    x = np.array(x + [x[0]])
    y = np.array(y + [y[0]])
    x, y = transform_to_img(x, y)
    ax.fill(x, y, c, alpha=0.5, zorder=3)


def _plot(x_path, y_path, th_path, ax):
    x_path = tf.concat(x_path, 0)
    y_path = tf.concat(y_path, 0)
    th_path = tf.concat(th_path, 0)
    path = tf.stack([x_path, y_path, th_path], axis=-1)
    cp = calculate_car_crucial_points(x_path, y_path, th_path)
    cp = tf.stack(cp, axis=1)
    _plot_car(cp[0, :, 0], ax, 'g')
    cl = ['c', 'g', 'b', 'm', 'k']
    for s in range(path.shape[0]):
        x = x_path[s]
        y = y_path[s]
        th = th_path[s]
        cps = calculate_car_crucial_points(x, y, th)
        for j, p in enumerate(cps):
            x, y = transform_to_img(p[:, 0], p[:, 1])
            ax.plot(x, y, color=cl[j], zorder=4)
    _plot_car(cp[-1, :, -1], ax, 'r')


def main(map_path, as_path, xd, yd, thd, ax):
    # 1. Get datasets
    bs = 128
    model_path = "./trained_model/best-26"
    map = read_map(map_path)[tf.newaxis]
    p0 = np.array([0.4, 0., 0., 0.], dtype=np.float32)[np.newaxis]
    pk = np.array([xd, yd, thd, 0.], dtype=np.float32)[np.newaxis]
    path = np.stack([p0, pk], axis=1)
    ddy0 = np.array([0.], dtype=np.float32)
    data = (map, path, ddy0)

    # 2. Define model
    model = PlanningNetworkMP(7, (bs, 6))

    # 3. Optimization

    optimizer = tf.keras.optimizers.Adam(1e-4)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(".", "", 1, model, optimizer)
    experiment_handler.restore(model_path)

    output, last_ddy = model(data, None, training=True)
    model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, _, x_path, y_path, th_path = plan_loss(
        output, data, last_ddy)
    print(model_loss)

    _plot(x_path, y_path, th_path, ax)

    uvc = np.loadtxt(as_path, delimiter="\t")
    u, v, c = np.split(uvc, 3, axis=-1)
    c = 181. * c
    ax.imshow(map[0, ..., 0], cmap='gray')
    return ax.scatter(u, v, c=c, s=1.5 * np.ones_like(c),
                zorder=3, cmap='hot_r')


if __name__ == '__main__':
    p = "../data/fig5/x/"
    c = range(14)
    map_path = [p + str(x) + ".png" for x in c]
    as_path = ["./fig5/as_grid_" + str(x) + ".tsv" for x in c]
    fig, axes = plt.subplots(nrows=2, ncols=7, gridspec_kw={"hspace": 0.015, "wspace": 0.015})
    #     1    2    3    4     5   6    7    8    9    10   11   12   13   14
    xs = [14., 19., 14., 16., 19., 19., 18., 15., 14., 18., 5., 19.5, 16., 19.]
    ys = [9.,  5.5, 9.5,  -2., 0.,  -5., 1.,  9., -3.5,  -1., 1., 0.5,  0.,  0.]
    ths = [np.pi/2, 0., np.pi/4., -np.pi/6, 0., -np.pi/10., 0., np.pi/3., -np.pi/20., np.pi/10, np.pi/6., np.pi/4., 0., -np.pi/6]
    #         1     2      3         4      5        6      7      8         9           10     11           12     13  14
    for i, ax in enumerate(axes.flat):
        ax.tick_params(
            axis='both',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft = False)
        im = main(map_path[i], as_path[i], xs[i], ys[i], ths[i], ax)
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical")
    plt.show()
