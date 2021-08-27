import inspect
import os
import sys
from time import time

import numpy as np
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import Loss, _plot, PlanningNetworkMP

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

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    # 1. Get datasets
    #bs = 128
    bs = 128 + 64
    #bs = 256
    #model_path = "./trained_9d/best-71"
    #model_path = "./trained_models/N=3_5e-4_monster/best-25"
    model_path = "./trained_models/N=3_5e-4_nodistloss/best-23"
    #a = np.loadtxt("/home/piotr/corl/grid.path").astype(np.float32)
    #p0 = a[:, 1:5]
    #pk = a[:, 5:]
    #path = np.stack([p0, pk], axis=1)
    for w in list(range(14)):
        ds, ds_size = scenarios.planning_dataset_grid(f"../data/exemplary_paths/x/{w}.png")

        ds = ds \
            .batch(bs) \
            .prefetch(bs)

        # 2. Define model
        N = 3
        model = PlanningNetworkMP(N)
        loss = Loss(N)

        # 3. Optimization

        optimizer = tf.keras.optimizers.Adam(1e-4)

        # 4. Restore, Log & Save
        experiment_handler = ExperimentHandler(".", "", 1, model, optimizer)
        experiment_handler.restore(model_path)

        acc = []
        xyth = []
        t0 = time()
        for i, data in _ds('Train', ds, ds_size, 0, bs):
            print(i)
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            output, _ = model(data, None, training=True)
            model_loss, invalid_loss, curvature_loss, dist_loss, tcurv, x_path, y_path, th_path, curvature = loss(output, data)

            # 5.1.3 Calculate statistics
            t = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
            s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))
            valid = tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32)
            acc.append(valid)
            p = data[1]
            xyth.append(p[:, -1, :3])
        t1 = time()
        print("RUNTIME:", t1 - t0)

        #f, g, h = 88, 88, 46
        #f, g, h = 23, 1, 1

        f, g, h = 128, 128, 46
        #f, g, h = 64, 64, 46
        xyth = tf.concat(xyth, axis=0)
        a = tf.reshape(xyth, (f, g, h, 3))
        b = tf.concat(acc, -1)
        b = tf.reshape(b, (f, g, h))
        xy = a[:, :, 0, :2]
        x = xy[..., 0]
        y = xy[..., 1]
        acc = tf.reduce_mean(b, axis=-1)
        res = 0.2
        u = -y / res + 64
        v = 120 - x / res
        color = acc
        #plt.imshow(data[0][0, ..., 0], cmap='gray')
        u = tf.reshape(u, [-1])[::-1]
        v = tf.reshape(v, [-1])[::-1]
        c = tf.reshape(color, [-1])[::-1]
        select = c != 0
        u = u[select]
        v = v[select]
        c = c[select]
        #plt.scatter(u, v, c=c, s=1.5 * np.ones_like(c),
        #            zorder=3, cmap='hot_r')
        #plt.colorbar(orientation="horizontal")
        #plt.show()
        epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))
        print(epoch_accuracy)
        uvc = tf.stack([u, v, c], axis=-1).numpy()
        np.savetxt(f"accessible_sets/as_grid_{w}.tsv", uvc, delimiter="\t")
        #np.savetxt("as_left_test_4.tsv", uvc, delimiter="\t")



if __name__ == '__main__':
    main()
