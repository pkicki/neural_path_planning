import inspect
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import PlanningNetworkMP, Loss
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from time import time

from utils.execution import ExperimentHandler

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
    bs = 1#28
    #model_path = "./trained_models/best-23"
    #model_path = "./trained_models/N=2_5e-4_nodistloss/best-16"
    #model_path = "./trained_models/N=3_5e-4_nodistloss/best-23"
    #model_path = "./trained_models/N=4_5e-4_nodistloss/best-32"
    #model_path = "./trained_models/N=5_5e-4_nodistloss/best-29"
    #model_path = "./trained_models/porthos/icra_task_map_1e-4_1/best-10"
    #model_path = "./trained_models/porthos/icra_task_map_1e-4/best-14"
    model_path = "./trained_models/porthos/icra_task_map_1e-4_CPU/best-18"
    #model_path = "./trained_models/aramis/icra_tm_N=4/best-26"
    ds_path = "../../data_new/test/all"
    # 1. Get datasets
    ds, ds_size = scenarios.planning_dataset(ds_path)

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

    # 5. Run everything
    acc = []
    times = []
    max_curvs = []
    for i, data in _ds('Check', ds, ds_size, 0, bs):
        start = time()
        output, cps = model(data, None, training=False)
        end = time()
        times.append(end - start)
        model_loss, invalid_loss, curvature_loss, cp_dists_loss, total_curvature_loss, x_path, y_path, th_path, curvature = loss(
            output, data)

        max_curvs.append(np.max(np.abs(curvature), axis=-1))
        valid = tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32)
        acc.append(valid)

    epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))
    max_curvs = tf.concat(max_curvs, -1)
    print("ACCURACY:", epoch_accuracy)
    print("MEAN PLANNING TIME:", np.mean(times[20:]))
    print("STD PLANNING TIME:", np.std(times[20:]))
    print("MEAN MAX CURVATURE:", np.mean(max_curvs))
    print("STD MAX CURVATURE:", np.std(max_curvs))



if __name__ == '__main__':
    main()
