import inspect
import os
import sys
from time import time

import numpy as np
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
#from dataset import scenarios
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
            yield (i, data)
            pbar.update(batch_size)

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    bs = 1
    model_path = "./trained_model/best-26"
    #ds_path = "../data/test/all"
    ds_path = "../data/at/"
    # 1. Get datasets
    ds, ds_size = scenarios.planning_dataset(ds_path)

    ds = ds \
        .batch(bs) \
        .prefetch(bs)

    # 2. Define model
    model = PlanningNetworkMP(7, (bs, 6))

    # 3. Optimization

    optimizer = tf.keras.optimizers.Adam(1e-4)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(".", "", 1, model, optimizer)
    experiment_handler.restore(model_path)

    # 5. Run everything
    acc = []
    nr = 0
    times = []
    for i, data in _ds('Check', ds, ds_size, 0, bs):
        map, path, ddy0 = data
        d = (map, path, ddy0)
        start = time()
        output, last_ddy = model(d, None, training=True)
        end = time()
        times.append(end - start)
        model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, _, x_path, y_path, th_path = plan_loss(output, d, last_ddy)


        ## 5.1.3 Calculate statistics
        t = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
        s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))
        u = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss + overshoot_loss, 0.0), tf.float32))
        valid = tf.cast(tf.equal(invalid_loss + curvature_loss + overshoot_loss, 0.0), tf.float32)
        acc.append(valid)

    epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))
    print("ACCURACY:", epoch_accuracy)
    print("MEAN PLANNING TIME:", np.mean(times[20:]))
    print("STD PLANNING TIME:", np.std(times[20:]))


if __name__ == '__main__':
    main()
