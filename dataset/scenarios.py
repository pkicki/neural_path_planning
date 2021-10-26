import os
from random import shuffle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# tf.enable_eager_execution()
def planning_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        task_map_path = scn_path[:-3] + "png"
        map_path = scn_path[:-7] + ".png"
        paths = np.load(scn_path)
        n_max = 256 + 128
        if paths.shape[0] > n_max:
            paths = paths[:n_max]
        else:
            pad_l = n_max - paths.shape[0]
            paths = np.pad(paths, ((0, pad_l), (0, 0)), mode='edge')
        ddy0 = paths[0, -1]
        return map_path, paths, task_map_path, ddy0

    def read_map(map_path, path, task_map_path, ddy0):
        img = tf.io.read_file(map_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        free = img > 0.5
        obs = img < 0.5
        img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
        task_img = tf.io.read_file(task_map_path)
        task_img = tf.io.decode_png(task_img, channels=1)
        task_img = tf.image.convert_image_dtype(task_img, tf.float32)
        return img, path, task_img, ddy0

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".npy")]
    scenarios = [(map_path, paths, task_map_path, ddy0) for map_path, paths, task_map_path, ddy0 in scenarios if len(paths)]

    g = list(range(len(scenarios)))
    shuffle(g)

    def gen():
        for i in g:
            yield scenarios[i]

    ds = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32, tf.string, tf.float32)) \
        .shuffle(buffer_size=int(0.1 * len(scenarios)), reshuffle_each_iteration=True).map(read_map, num_parallel_calls=8)

    return ds, len(scenarios)

def planning_dataset_grid(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        # map = plt.imread(scn_path)[..., :1]
        res_path = scn_path[:-3] + "path"
        paths = []
        ddy0s = []
        # print(res_path)
        with open(res_path, 'r') as fh:
            lines = fh.read().split('\n')[:-1]
            for i, l in enumerate(lines):
                xythk = np.array(l.split()).astype(np.float32)
                ddy0 = xythk[0]
                xythk = np.reshape(xythk[1:], (-1, 4))
                paths.append(xythk)
                ddy0s.append(ddy0)
        ddy0s = np.array(ddy0s).astype(np.float32)
        paths = np.stack(paths, 0).astype(np.float32)
        return scn_path, paths, ddy0s

    def read_map(map_path, path, ddy0):
        img = tf.io.read_file(map_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        free = img > 0.5
        obs = img < 0.5
        img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
        return img, path, ddy0

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".png")]

    g = list(range(len(scenarios)))

    def gen():
        for i in g:
            s = list(range(len(scenarios[i][1])))
            for k in s:
                yield scenarios[i][0], scenarios[i][1][k], scenarios[i][2][k]

    ds = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32, tf.float32)) \
        .map(read_map, num_parallel_calls=8)

    return ds, len(scenarios)


def carla_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        p0 = np.array([0.4, 0., 0.])
        pk = np.array([15., 9., np.pi/2.])
        paths = np.stack([p0, pk])[tf.newaxis]
        ddy0s = np.zeros((1,))
        return scn_path, paths, ddy0s

    def read_map(map_path, path, ddy0):
        img = tf.io.read_file(map_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        free = img > 0.5
        obs = img < 0.5
        img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
        return img, path, map_path, 0, ddy0

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if "map" in f]

    g = list(range(len(scenarios)))

    def gen():
        for i in g:
            s = list(range(len(scenarios[i][1])))
            for k in s:
                yield scenarios[i][0], scenarios[i][1][k], scenarios[i][2][k]

    ds = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32, tf.float32)) \
        .shuffle(buffer_size=int(1 * len(scenarios)), reshuffle_each_iteration=True).map(read_map, num_parallel_calls=8)

    return ds, len(scenarios)
