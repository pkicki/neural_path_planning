from math import pi
from time import time

import tensorflow as tf
import numpy as np

from utils.constants import Car
from utils.crucial_points import transform_points
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
        # CoordConv
        # x_indices, y_indices = tf.meshgrid(2 * (tf.range(x.shape[1]) / (x.shape[1] - 1)) - 1,
        #                                     2 * (tf.range(x.shape[2]) / (x.shape[2] - 1)) - 1)
        # xy_indices = tf.tile(tf.stack([x_indices , y_indices], axis=-1)[tf.newaxis], (bs, 1, 1, 1))
        # xy_indices = tf.cast(xy_indices, tf.float32)
        # x = tf.concat([x, xy_indices], axis=-1)
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, -1))
        for layer in self.fc:
            x = layer(x)
        return x


def unpack_data(data):
    map, path, task_map, sdf = data
    p0 = path[:, 0]
    x0, y0, th0, beta0 = tf.unstack(p0, axis=-1)
    pk = path[:, -1]
    xk, yk, thk, betak = tf.unstack(pk, axis=-1)
    return map, path, task_map, sdf, x0, y0, th0, beta0, xk, yk, thk, betak


class PlanningNetworkMP(tf.keras.Model):
    def __init__(self, depth):
        super(PlanningNetworkMP, self).__init__()

        n = 256
        self.depth = depth
        self.n = 7
        self.n_pts = 3 + (2 ** depth - 1) + 2
        self.m = self.n + self.n_pts

        self.map_processing = MapFeaturesProcessor(n)
        self.preprocessing_stage = FeatureExtractorLayer(n)
        self.pts_est = EstimatorLayer((2 ** self.depth - 1) * 2 + 3)
        self.dim = 25.6

    def call(self, data, env_features, training=None):
        map, path, task_map, sdf, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)

        env_representation = tf.concat([sdf, task_map], axis=-1)
        env_features = self.map_processing(env_representation)

        inputs = tf.stack([x0 / self.dim, y0 / (self.dim / 2), np.sin(th0), np.cos(th0), beta0,
                           xk / self.dim, yk / (self.dim / 2), np.sin(thk), np.cos(thk)], -1)

        features = self.preprocessing_stage(inputs, training)
        features = tf.concat([features, env_features], -1)

        p = self.pts_est(features, training)
        pts = p[..., :-3]
        x1x2r = p[..., -3:]
        pts = tf.reshape(pts, (-1, 2 ** self.depth - 1, 2))
        return self.calculate_control_points(data, pts, x1x2r), pts, p

    def calculate_control_points(self, data, pts, x1x2r):
        _, _, _, _, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)
        # move data to (0;1) and (-0.5; 0.5) for x and y respectively
        x1x2r = (x1x2r + 1) / 2.
        x0 /= self.dim
        y0 /= self.dim / 2
        xk /= self.dim
        yk /= self.dim / 2
        kappa0 = 1 / Car.L * tf.tan(beta0)
        # x1 = x0 + 0.03 * x1x2r[:, 0] + 1e-5
        # x2 = x1 + 0.03 * x1x2r[:, 1] + 1e-5
        x1 = x0 + 0.1 * x1x2r[:, 0] + 1e-5
        x2 = x1 + 0.1 * x1x2r[:, 1] + 1e-5
        y1 = tf.zeros_like(x1)
        # y2 = 3 * kappa0 * (x1 - x0) ** 2
        y2 = kappa0 * (x1 - x0) ** 2 / (
                    (self.m - 2 * self.n) * self.n * self.n * (self.m - 2 * self.n) ** 2 * (self.n - 1) / 2)
        # r = 0.015
        # r = 0.03 * x1x2r[:, 2] + 1e-5
        r = 0.1 * x1x2r[:, 2] + 1e-5
        xkm1 = xk - r * tf.cos(thk)
        ykm1 = yk - r * tf.sin(thk) * 2  # TODO: crazy shit!!!!!
        xy0 = tf.stack([x0, y0], axis=-1)[:, tf.newaxis]
        xy1 = tf.stack([x1, y1], axis=-1)[:, tf.newaxis]
        xy2 = tf.stack([x2, y2], axis=-1)[:, tf.newaxis]
        xykm1 = tf.stack([xkm1, ykm1], axis=-1)[:, tf.newaxis]
        xyk = tf.stack([xk, yk], axis=-1)[:, tf.newaxis]

        # def middle(lb, ub, pred):
        #    v = ub - lb
        #    v_p = tf.stack([-v[:, :, 1], v[:, :, 0]], axis=-1)
        #    p1 = (pred[..., :1] + 1) / 2
        #    p2 = pred[..., 1:]
        #    new = lb + v * p1 + v_p * p2
        #    return new

        def middle(lb, ub, pred):
            mid = (lb + ub) / 2
            diff = 0.5 * tf.reduce_max(tf.abs(lb - ub), axis=-1, keepdims=True)
            old = mid + pred * diff
            return old

        f = [xy2, xykm1]
        p = 0
        for d in range(self.depth):
            i = 0
            while i < len(f) - 1:
                mid = middle(f[i], f[i + 1], pts[:, p, tf.newaxis])
                f.insert(i + 1, mid)
                i += 2
                p += 1
        cp = tf.concat([xy0, xy1] + f + [xyk], axis=1)
        return cp


class DummyPlanner(tf.keras.Model):
    def __init__(self, first, depth):
        super(DummyPlanner, self).__init__()
        self.pts = tf.Variable(first, trainable=True)
        self.dim = 25.6
        self.depth = depth
        self.n = 7
        self.n_pts = 3 + (2 ** depth - 1) + 2
        self.m = self.n + self.n_pts

    def call(self, data):
        pts = self.pts[..., :-3]
        x1x2r = self.pts[..., -3:]
        pts = tf.reshape(pts, (-1, 2 ** self.depth - 1, 2))
        return self.calculate_control_points(data, pts, x1x2r), pts

    def calculate_control_points(self, data, pts, x1x2r):
        _, _, _, _, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)
        x1x2r = (x1x2r + 1) / 2.
        # move data to (0;1) and (-0.5; 0.5) for x and y respectively
        x0 /= self.dim
        y0 /= self.dim / 2
        xk /= self.dim
        yk /= self.dim / 2
        kappa0 = 1 / Car.L * tf.tan(beta0)
        x1 = x0 + 0.1 * x1x2r[:, 0] + 1e-5
        x2 = x1 + 0.1 * x1x2r[:, 1] + 1e-5
        y1 = tf.zeros_like(x1)
        # y2 = 3 * kappa0 * (x1 - x0) ** 2
        y2 = kappa0 * (x1 - x0) ** 2 / (
                    (self.m - 2 * self.n) * self.n * self.n * (self.m - 2 * self.n) ** 2 * (self.n - 1) / 2)
        r = 0.1 * x1x2r[:, 2] + 1e-5
        xkm1 = xk - r * tf.cos(thk)
        ykm1 = yk - r * tf.sin(thk) * 2  # TODO: crazy shit!!!!!
        xy0 = tf.stack([x0, y0], axis=-1)[:, tf.newaxis]
        xy1 = tf.stack([x1, y1], axis=-1)[:, tf.newaxis]
        xy2 = tf.stack([x2, y2], axis=-1)[:, tf.newaxis]
        xykm1 = tf.stack([xkm1, ykm1], axis=-1)[:, tf.newaxis]
        xyk = tf.stack([xk, yk], axis=-1)[:, tf.newaxis]

        def middle(lb, ub, pred):
            mid = (lb + ub) / 2
            diff = 0.5 * tf.reduce_max(tf.abs(lb - ub), axis=-1, keepdims=True)
            return mid + pred * diff

        f = [xy2, xykm1]
        p = 0
        for d in range(self.depth):
            i = 0
            while i < len(f) - 1:
                mid = middle(f[i], f[i + 1], pts[:, p, tf.newaxis])
                f.insert(i + 1, mid)
                i += 2
                p += 1
        cp = tf.concat([xy0, xy1] + f + [xyk], axis=1)
        return cp


class Loss:
    def __init__(self, depth):
        self.n = 7
        self.n_pts = 3 + (2 ** depth - 1) + 2
        self.m = self.n + self.n_pts
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

        T = np.linspace(0., 1., 1024)
        Ns = [np.stack([N(self.n, t, i) for i in range(self.m - self.n)]) for t in T]
        Ns = np.stack(Ns, axis=0)
        Ns[-1, -1] = 1.
        dNs = [np.stack([dN(self.n, t, i) for i in range(self.m - self.n)]) for t in T]
        dNs = np.stack(dNs, axis=0)
        dNs[-1, -1] = (self.m - 2 * self.n) * self.n
        dNs[-1, -2] = -(self.m - 2 * self.n) * self.n
        ddNs = [np.stack([ddN(self.n, t, i) for i in range(self.m - self.n)]) for t in T]
        ddNs = np.stack(ddNs, axis=0)
        ddNs[-1, -1] = 2 * self.n * (self.m - 2 * self.n) ** 2 * (self.n - 1) / 2
        ddNs[-1, -2] = -3 * self.n * (self.m - 2 * self.n) ** 2 * (self.n - 1) / 2
        ddNs[-1, -3] = self.n * (self.m - 2 * self.n) ** 2 * (self.n - 1) / 2
        return Ns[np.newaxis], dNs[np.newaxis], ddNs[np.newaxis]

    def __call__(self, plan, data):
        map, path, task_map, sdf, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)
        x_plan = plan[:, :, 0] * self.dim
        y_plan = plan[:, :, 1] * (self.dim / 2.)
        plan_g = tf.stack([x_plan, y_plan], axis=-1)
        xy = self.N @ plan_g
        dxy = self.dN @ plan_g
        ddxy = self.ddN @ plan_g
        curvature = (ddxy[..., 1] * dxy[..., 0] - ddxy[..., 0] * dxy[..., 1]) / \
                    (tf.reduce_sum(tf.square(dxy), axis=-1) + 1e-8) ** (3. / 2)
        tcurv = tf.reduce_sum(tf.abs(curvature[:, 1:] - curvature[:, :-1]), axis=-1)
        curvature_loss = tf.reduce_sum(tf.nn.relu(tf.abs(curvature) - Car.max_curvature), -1)
        x_global = xy[..., 0]
        y_global = xy[..., 1]
        th_global = tf.atan2(dxy[..., 1], dxy[..., 0])
        invalid_loss, supervised_loss = invalidate(x_global, y_global, th_global, map, path)
        coarse_loss = invalid_loss + curvature_loss
        fine_loss = invalid_loss + curvature_loss + 1e-1 * tcurv
        loss = tf.where(coarse_loss == 0, fine_loss, coarse_loss)
        return loss, invalid_loss, curvature_loss, curvature_loss, tcurv, x_global, y_global, th_global, curvature

    def curvature(self, plan):
        x_plan = plan[:, :, 0] * self.dim
        y_plan = plan[:, :, 1] * (self.dim / 2.)
        plan_g = tf.stack([x_plan, y_plan], axis=-1)
        xy = self.N @ plan_g
        dxy = self.dN @ plan_g
        ddxy = self.ddN @ plan_g
        curvature = (ddxy[..., 1] * dxy[..., 0] - ddxy[..., 0] * dxy[..., 1]) / \
                    (tf.reduce_sum(tf.square(dxy), axis=-1) + 1e-8) ** (3. / 2)
        curvature_loss = tf.reduce_sum(tf.nn.relu(tf.abs(curvature) - Car.max_curvature), -1)
        x_global = xy[..., 0]
        y_global = xy[..., 1]
        th_global = tf.atan2(dxy[..., 1], dxy[..., 0])
        return curvature_loss, x_global, y_global, th_global, curvature

    def curvature_loss(self, plan):
        plan = np.reshape(plan, (1, 12, 2))
        x_plan = plan[:, :, 0] * self.dim
        y_plan = plan[:, :, 1] * (self.dim / 2.)
        plan_g = np.stack([x_plan, y_plan], axis=-1)
        xy = self.N @ plan_g
        dxy = self.dN @ plan_g
        ddxy = self.ddN @ plan_g
        curvature = (ddxy[..., 1] * dxy[..., 0] - ddxy[..., 0] * dxy[..., 1]) / \
                    (np.sum(tf.square(dxy), axis=-1) + 1e-8) ** (3. / 2)
        curvature_loss = np.sum(np.maximum(tf.abs(curvature) - Car.max_curvature, 0.), -1)
        return curvature_loss


def _plot(x_path, y_path, th_path, data, step, cps, idx=0, print=False):
    res = 0.2
    map, path, _, sdf = data
    path = path[..., :3]
    plt.imshow(map[idx, ..., 0])
    x = x_path[idx]
    y = y_path[idx]
    th = th_path[idx]

    cp = Car.crucial_points[np.newaxis]
    R = Rot(th)
    xy = np.stack([x, y], axis=-1)[:, np.newaxis]
    cp = transform_points(xy, R, cp)
    for p in cp:
        u = -p[:, 1] / res + 64
        v = 120 - p[:, 0] / res
        plt.plot(u, v)
        # plt.plot(p[:, 0] * 10, (10. - p[:, 1]) * 10)
    # plt.plot(path[0, :, 0] * 10, (10 - path[0, :, 1]) * 10, 'r')
    u = -path[idx, :, 1] / res + 64
    v = 120 - path[idx, :, 0] / res
    plt.plot(u, v, 'r')
    cps_u = -cps[idx, :, 1] * 12.8 / res + 64
    cps_v = 120 - cps[idx, :, 0] * 25.6 / res
    # plt.plot(25.6 * cps[0, :, 0],  12.8 * cps[0, :, 1], 'bx')
    plt.plot(cps_u, cps_v, 'bx')
    ax = plt.gca()
    for i in range(len(cps_u)):
        ax.annotate(str(i), (cps_u[i], cps_v[i]))
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


def invalidate(x, y, fi, free_space, path):
    """
        Check how much specified points violate the environment constraints
    """
    cp = Car.crucial_points[np.newaxis, np.newaxis]
    R = Rot(fi)
    xy = np.stack([x, y], axis=-1)[:, :, np.newaxis]
    crucial_points = transform_points(xy, R, cp)

    car_contour = Car.contour[np.newaxis, np.newaxis]
    car_contour = transform_points(xy, R, car_contour)

    d = tf.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)

    path_cp = transform_points(path[..., np.newaxis, :2], Rot(path[..., 2]), cp)
    penetration = path_dist_cp(path_cp, crucial_points)
    not_in_collision = if_inside(free_space, car_contour)
    not_in_collision = tf.reduce_all(not_in_collision, axis=-1)
    penetration = tf.where(not_in_collision, tf.zeros_like(penetration), penetration)

    violation_level = tf.reduce_sum(d[..., 0] * penetration[:, :-1], -1)
    supervised_loss = tf.reduce_sum(penetration, -1)
    return violation_level, supervised_loss
