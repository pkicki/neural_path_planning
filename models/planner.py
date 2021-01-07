from math import pi
from time import time

import tensorflow as tf
import numpy as np

from utils.constants import Car
from utils.crucial_points import calculate_car_crucial_points, calculate_car_body, calculate_car_contour
from utils.distances import dist, path_dist, if_inside, path_line_dist, path_dist_cp
from utils.poly5 import curvature, params
from utils.utils import _calculate_length, Rot
from matplotlib import pyplot as plt


class EstimatorLayer(tf.keras.Model):
    """
    Parameter estimator layer
    """

    def __init__(self, m):
        super(EstimatorLayer, self).__init__()
        self.features = [
            tf.keras.layers.Dense(256, tf.nn.tanh),
            tf.keras.layers.Dense(128, tf.nn.tanh),
            tf.keras.layers.Dense(m, tf.nn.tanh),
        ]

    def call(self, x, training=None):
        for layer in self.features:
            x = layer(x)
        return x


class FeatureExtractorLayer(tf.keras.Model):
    """
    Feature exrtactor layer
    """

    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(FeatureExtractorLayer, self).__init__()
        self.features = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(num_features, activation),
            tf.keras.layers.Dense(num_features, activation),
            tf.keras.layers.Dense(num_features, activation),
        ]

    def call(self, x, training=None):
        for layer in self.features:
            x = layer(x)
        return x


class MapFeaturesProcessor(tf.keras.Model):
    def __init__(self, num_features):
        super(MapFeaturesProcessor, self).__init__()
        self.features = [
            tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation=tf.keras.activations.relu),
        ]

        self.fc = [
            tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(num_features, activation=tf.keras.activations.tanh),
        ]

    def call(self, x, training=None):
        bs = x.shape[0]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, -1))
        for layer in self.fc:
            x = layer(x)
        return x


def unpack_data(data):
    map, path = data
    p0 = path[:, 0]
    x0, y0, th0, beta0 = tf.unstack(p0, axis=-1)
    pk = path[:, -1]
    xk, yk, thk, betak = tf.unstack(pk, axis=-1)
    return map, path, x0, y0, th0, beta0, xk, yk, thk, betak


class PlanningNetworkMP(tf.keras.Model):

    def __init__(self, n_pts):
        super(PlanningNetworkMP, self).__init__()

        n = 256
        self.n_pts = n_pts

        self.map_processing = MapFeaturesProcessor(n)
        self.preprocessing_stage = FeatureExtractorLayer(n)
        self.pts_est = EstimatorLayer((self.n_pts - 1) * 2)
        self.dim = 25.6

    def call(self, data, map_features, training=None):
        map, path, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)

        map_features = self.map_processing(map)

        inputs = tf.stack([x0 / self.dim, y0 / (self.dim / 2), np.sin(th0), np.cos(th0), beta0,
                           xk / self.dim, yk / (self.dim / 2), np.sin(thk), np.cos(thk)], -1)

        features = self.preprocessing_stage(inputs, training)
        features = tf.concat([features, map_features], -1)

        p = self.pts_est(features, training)
        p = tf.reshape(p, (-1, self.n_pts - 1, 2))
        z = tf.zeros_like(p[:, :1, 0])
        o = -tf.ones_like(p[:, :1, 0])
        oz = tf.stack([o, z], axis=-1)
        pts = tf.concat([oz, p], axis=-2)
        return pts


class Placeholder(tf.keras.Model):

    def __init__(self, n_pts):
        super(Placeholder, self).__init__()
        self.p = tf.Variable(np.random.uniform(-1., 1., (1, n_pts - 1, 2)), trainable=True, dtype=tf.float32)


    def call(self, data, map_features, training=None):
        p = self.p
        z = tf.zeros_like(p[:, :1, 0])
        o = -tf.ones_like(p[:, :1, 0])
        oz = tf.stack([o, z], axis=-1)
        pts = tf.concat([oz, p], axis=-2)
        return pts

class Loss:
    def __init__(self, n_pts):
        self.T = np.linspace(0., 1., 512)
        self.n = 3
        self.m = 3 + n_pts
        self.u = np.pad(np.linspace(0., 1., self.m + 1 - 2 * self.n), self.n, 'edge')
        self.N, self.dN, self.ddN = self.calculate_N()
        self.dim = 25.6

    def calculate_N(self):
        def N(n, t, i):
            if n == 0:
                if self.u[i] <= t < self.u[i + 1]:
                    return 1
                else:
                    return 0
            s = 0.
            if self.u[i + n] - self.u[i] != 0:
                s += (t - self.u[i]) / (self.u[i + n] - self.u[i]) * N(n - 1, t, i)
            if self.u[i + n + 1] - self.u[i + 1] != 0:
                s += (self.u[i + n + 1] - t) / (self.u[i + n + 1] - self.u[i + 1]) * N(n - 1, t, i + 1)
            return s

        def dN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.
            if m1 != 0:
                s += N(n - 1, t, i) / m1
            if m2 != 0:
                s -= N(n - 1, t, i + 1) / m2
            return n * s

        def ddN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.
            if m1 != 0:
                s += dN(n - 1, t, i) / m1
            if m2 != 0:
                s -= dN(n - 1, t, i + 1) / m2
            return n * s

        Ns = [np.stack([N(self.n, t, i) for i in range(self.m - self.n)]) for t in self.T]
        Ns = np.stack(Ns, axis=0)
        Ns[-1, -1] = 1.
        dNs = [np.stack([dN(self.n, t, i) for i in range(self.m - self.n)]) for t in self.T]
        dNs = np.stack(dNs, axis=0)
        dNs[-1, -1] = 21.
        dNs[-1, -2] = -21.
        ddNs = [np.stack([ddN(self.n, t, i) for i in range(self.m - self.n)]) for t in self.T]
        ddNs = np.stack(ddNs, axis=0)
        ddNs[-1, -1] = 294.
        ddNs[-1, -2] = -441.
        ddNs[-1, -3] = 147.
        return Ns[np.newaxis], dNs[np.newaxis], ddNs[np.newaxis]

    def __call__(self, plan, data):
        map, path, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)
        v_plan = (plan[:, :, 0] + 1) * 5  # * self.dim
        z_plan = plan[:, :, 1]  # * (self.dim / 2.)
        plan_g = tf.stack([v_plan, z_plan], axis=-1)
        vz = self.N @ plan_g

        plt.subplot(121)
        plt.plot(self.T, vz[0, ..., 0], label="v")
        plt.plot(self.T, vz[0, ..., 1], label="zeta")
        plt.legend()

        state0 = tf.stack([x0, y0, th0, beta0], axis=-1)
        Tp = 10. / len(self.T)
        states = [state0]
        for i in range(self.T.shape[0]):
            s = states[-1]
            zeros = tf.zeros_like(s[..., -1])
            ones = tf.ones_like(s[..., -1])
            G1 = tf.stack([tf.cos(s[..., -2]), tf.sin(s[..., -2]), tf.tan(s[..., -1]) / Car.L, zeros], axis=-1)
            G2 = tf.stack([zeros, zeros, zeros, ones], axis=-1)
            G = tf.stack([G1, G2], axis=-1)
            ns = s + (Tp * G @ vz[:, i, :, tf.newaxis])[..., 0]
            x, y, th, beta = tf.split(ns, 4, axis=-1)
            beta = tf.clip_by_value(beta, -Car.max_beta, Car.max_beta)
            ns = tf.concat([x, y, th, beta], axis=-1)
            states.append(ns)
        states = tf.stack(states, axis=-2)[:, 1:]
        x_global = states[..., 0]
        y_global = states[..., 1]
        th_global = states[..., 2]
        beta_global = states[..., 3]
        curvature_loss = tf.reduce_mean(tf.nn.relu(beta_global - Car.max_beta), axis=-1)
        plt.subplot(122)
        plt.ylim(-self.dim / 2, self.dim / 2)
        plt.xlim(0., self.dim)
        plt.plot(x_global[0], y_global[0], label="xy")
        plt.legend()
        #plt.savefig("./last.png")
        plt.show()
        plt.clf()
        dx = tf.nn.relu(tf.abs(xk - x_global[:, -1]) - 0.2)
        dy = tf.nn.relu(tf.abs(yk - y_global[:, -1]) - 0.2)
        dth = 10 * tf.nn.relu(tf.abs(thk - th_global[:, -1]) - 0.05)
        overshoot_loss = dx + dy + dth

        outside_loss_x = tf.nn.relu(x_global - self.dim) + tf.nn.relu(-x_global)
        outside_loss_y = tf.nn.relu(abs(y_global) - self.dim / 2)
        outside_loss = tf.reduce_sum(outside_loss_x + outside_loss_y, axis=-1)
        invalid_loss, supervised_loss = invalidate(x_global, y_global, th_global, map, path)
        # loss = invalid_loss + curvature_loss
        coarse_loss = invalid_loss + 1e-3 * curvature_loss + outside_loss + overshoot_loss
        # fine_loss = invalid_loss + 1e-3 * curvature_loss + 1e-3 * tcurv
        # loss = tf.where(coarse_loss == 0, fine_loss, coarse_loss)
        loss = coarse_loss
        return loss, invalid_loss, curvature_loss, overshoot_loss, x_global, y_global, th_global


def _plot(x_path, y_path, th_path, data, step, cps, print=False):
    res = 0.2
    map, path = data
    path = path[..., :3]
    plt.imshow(map[0, ..., 0])
    x = x_path[0]
    y = y_path[0]
    th = th_path[0]
    cp = calculate_car_crucial_points(x, y, th)
    for p in cp:
        u = -p[:, 1] / res + 64
        v = 120 - p[:, 0] / res
        plt.plot(u, v)
        # plt.plot(p[:, 0] * 10, (10. - p[:, 1]) * 10)
    # plt.plot(path[0, :, 0] * 10, (10 - path[0, :, 1]) * 10, 'r')
    u = -path[0, :, 1] / res + 64
    v = 120 - path[0, :, 0] / res
    plt.plot(u, v, 'r')
    cps_u = -cps[0, :, 1] * 12.8 / res + 64
    cps_v = 120 - cps[0, :, 0] * 25.6 / res
    # plt.plot(25.6 * cps[0, :, 0],  12.8 * cps[0, :, 1], 'bx')
    plt.plot(cps_u, cps_v, 'bx')
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


def invalidate(x, y, fi, free_space, path):
    """
        Check how much specified points violate the environment constraints
    """
    crucial_points = calculate_car_crucial_points(x, y, fi)
    car_contour = calculate_car_contour(crucial_points)
    crucial_points = tf.stack(crucial_points, -2)
    xy = tf.stack([x, y], axis=-1)[:, :, tf.newaxis]

    d = tf.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)

    path_cp = calculate_car_crucial_points(path[..., 0], path[..., 1], path[..., 2])
    path_cp = tf.stack(path_cp, -2)
    penetration = path_dist_cp(path_cp, crucial_points)
    not_in_collision = if_inside(free_space, car_contour)
    not_in_collision = tf.reduce_all(not_in_collision, axis=-1)
    penetration = tf.where(not_in_collision, tf.zeros_like(penetration), penetration)

    violation_level = tf.reduce_sum(d[..., 0] * penetration[:, :-1], -1)

    supervised_loss = tf.reduce_sum(penetration, -1)
    return violation_level, supervised_loss
