import os
from random import shuffle, random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# tf.enable_eager_execution()

def planning_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        map_path = scn_path[:-4] + "png"
        sdf_path = scn_path[:-5] + "_sdf_free.png"
        # map = plt.imread(scn_path)[..., :1]
        paths = []
        # print(res_path)
        with open(scn_path, 'r') as fh:
            lines = fh.read().split('\n')[:-1]
            for i, l in enumerate(lines):
                xythk = np.array(l.split()).astype(np.float32)
                xythk = np.reshape(xythk[1:], (-1, 4))
                n_max = 256 + 128
                # n_max = 512
                if xythk.shape[0] > n_max:
                    xythk = xythk[:n_max]
                else:
                    pad_l = n_max - xythk.shape[0]
                    xythk = np.pad(xythk, ((0, pad_l), (0, 0)), mode='edge')
                x0 = xythk[0, 0]
                y0 = xythk[0, 1]
                xk = xythk[-1, 0]
                yk = xythk[-1, 1]
                if np.sqrt((x0 - xk) ** 2 + (y0 - yk) ** 2) > 5:
                    paths.append(xythk)
        if paths:
            paths = np.stack(paths, 0).astype(np.float32)
        return map_path, sdf_path, paths

    def read_map(map_path, sdf_path, path):
        img = tf.io.read_file(map_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        free = img > 0.5
        obs = img < 0.5
        img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
        sdf = tf.io.read_file(sdf_path)
        sdf = tf.io.decode_png(sdf, channels=1)
        sdf = tf.image.convert_image_dtype(sdf, tf.float32)
        return img, sdf, path

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".path")]
    scenarios = [(map_path, sdf_path, paths) for map_path, sdf_path, paths in scenarios if len(paths)]

    g = list(range(len(scenarios)))
    shuffle(g)

    def gen():
        for i in g:
            s = list(range(len(scenarios[i][2])))
            shuffle(s)
            for k in s:
                # if random() > 0.5:
                #    yield scenarios[i][0], scenarios[i][1][k]
                # else:
                #    a = scenarios[i][0].replace(".png", "_r.png")
                #    path = scenarios[i][1][k]
                #    x = path[:, 0]
                #    y = -path[:, 1]
                #    th = -path[:, 2]
                #    beta = -path[:, 3]
                #    b = tf.stack([x, y, th, beta], axis=-1)
                #    yield a, b
                yield scenarios[i][0], scenarios[i][1], scenarios[i][2][k]

    ds = tf.data.Dataset.from_generator(gen, (tf.string, tf.string, tf.float32)) \
        .shuffle(buffer_size=int(1 * len(scenarios)), reshuffle_each_iteration=True).map(read_map, num_parallel_calls=8)

    return ds, len(scenarios)


def planning_dataset_grid(path):
    def read_scn(scn_path):
        # map = plt.imread(scn_path)[..., :1]
        # res_path = scn_path[:-3] + "path"
        path = "/".join(scn_path.split("/")[:-1])
        res_path = os.path.join(path, "test.path")
        paths = []
        # print(res_path)
        with open(res_path, 'r') as fh:
            lines = fh.read().split('\n')[:-1]
            for i, l in enumerate(lines):
                xythk = np.array(l.split()).astype(np.float32)
                xythk = np.reshape(xythk[1:], (-1, 4))
                paths.append(xythk)
        paths = np.stack(paths, 0).astype(np.float32)
        return scn_path, paths

    def read_map(map_path, path):
        img = tf.io.read_file(map_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        free = img > 0.5
        obs = img < 0.5
        img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
        return img, path

    scenarios = [read_scn(path)]

    g = list(range(len(scenarios)))

    def gen():
        for i in g:
            s = list(range(len(scenarios[i][1])))
            for k in s:
                yield scenarios[i][0], scenarios[i][1][k]

    ds = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32)) \
        .map(read_map, num_parallel_calls=8)

    return ds, len(scenarios)


def carla_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        p0 = np.array([0.4, 0., 0.])
        pk = np.array([15., 9., np.pi / 2.])
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


def planning_dataset_validate(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        map_path = scn_path[:-4] + "png"
        # map = plt.imread(scn_path)[..., :1]
        paths = []
        # print(res_path)
        with open(scn_path, 'r') as fh:
            lines = fh.read().split('\n')[:-1]
            for i, l in enumerate(lines):
                xythk = np.array(l.split()).astype(np.float32)
                xythk = np.reshape(xythk[1:], (-1, 4))
                n_max = 256 + 128
                # n_max = 512
                if xythk.shape[0] > n_max:
                    xythk = xythk[:n_max]
                else:
                    pad_l = n_max - xythk.shape[0]
                    xythk = np.pad(xythk, ((0, pad_l), (0, 0)), mode='edge')
                x0 = xythk[0, 0]
                y0 = xythk[0, 1]
                xk = xythk[-1, 0]
                yk = xythk[-1, 1]
                if np.sqrt((x0 - xk) ** 2 + (y0 - yk) ** 2) > 5:
                    paths.append(xythk)
        if paths:
            paths = np.stack(paths, 0).astype(np.float32)
        return map_path, paths

    def read_map(map_path, path):
        img = tf.io.read_file(map_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        free = img > 0.5
        obs = img < 0.5
        img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
        return img, path, map_path

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".path")]
    scenarios = [(scn_path, paths) for scn_path, paths in scenarios if len(paths)]

    g = list(range(len(scenarios)))
    shuffle(g)

    def gen():
        for i in g:
            s = list(range(len(scenarios[i][1])))
            shuffle(s)
            for k in s:
                # if random() > 0.5:
                #    yield scenarios[i][0], scenarios[i][1][k]
                # else:
                #    a = scenarios[i][0].replace(".png", "_r.png")
                #    path = scenarios[i][1][k]
                #    x = path[:, 0]
                #    y = -path[:, 1]
                #    th = -path[:, 2]
                #    beta = -path[:, 3]
                #    b = tf.stack([x, y, th, beta], axis=-1)
                #    yield a, b
                yield scenarios[i][0], scenarios[i][1][k]

    ds = tf.data.Dataset.from_generator(gen, (tf.string, tf.float32)) \
        .shuffle(buffer_size=int(1 * len(scenarios)), reshuffle_each_iteration=True).map(read_map, num_parallel_calls=8)

    return ds, len(scenarios)
