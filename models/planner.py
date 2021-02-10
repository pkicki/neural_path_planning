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
        self.pts_est = EstimatorLayer(2 + (self.n_pts - 5)*2 + 1)
        self.dim = 25.6

    def call(self, data, map_features, training=None):
        map, path, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)

        map_features = self.map_processing(map)

        inputs = tf.stack([x0 / self.dim, y0 / (self.dim / 2), np.sin(th0), np.cos(th0), beta0,
                           xk / self.dim, yk / (self.dim / 2), np.sin(thk), np.cos(thk)], -1)

        features = self.preprocessing_stage(inputs, training)
        features = tf.concat([features, map_features], -1)

        p = self.pts_est(features, training)
        #p = tf.zeros_like(p)
        # bound the values for x1 and x2
        x1 = (p[:, 0] + 1.) / 50. + x0 / self.dim + 0.005
        x2 = (p[:, 1] + 1.) / 50. + x1 + 0.005
        pts = p[:, 2:-1]
        pts = tf.reshape(pts, (-1, self.n_pts - 5,  2))
        # make x lie in (-0.5, 1.5) and y in (-1., 1)
        xs = pts[..., 0] #+ 1.
        ys = pts[..., 1]
        pts = tf.stack([xs, ys], axis=-1)
        # define distance between last and one before last points
        r = (p[:, -1] + 1.) / 50. + 0.005
        return self.calculate_control_points(data, x1, x2, pts, r)

    def calculate_control_points(self, data, x1, x2, pts, r):
        _, _, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)
        # move data to (0;1) and (-0.5; 0.5) for x and y respectively
        x0 /= self.dim
        y0 /= self.dim / 2
        xk /= self.dim
        yk /= self.dim / 2
        kappa0 = 1 / Car.L * tf.tan(beta0)
        y1 = tf.zeros_like(x1)
        y2 = 3 * kappa0 * (x1 - x0)**2
        xkm1 = xk - r * tf.cos(thk)
        ykm1 = yk - r * tf.sin(thk)
        xy0 = tf.stack([x0, y0], axis=-1)[:, tf.newaxis]
        xy1 = tf.stack([x1, y1], axis=-1)[:, tf.newaxis]
        xy2 = tf.stack([x2, y2], axis=-1)[:, tf.newaxis]
        xykm1 = tf.stack([xkm1, ykm1], axis=-1)[:, tf.newaxis]
        xyk = tf.stack([xk, yk], axis=-1)[:, tf.newaxis]

        t = np.linspace(0., 1., self.n_pts - 3)[np.newaxis, 1:-1]
        mid_x = x0[:, np.newaxis] * (1 - t) + xk[:, np.newaxis] * t
        mid_y = y0[:, np.newaxis] * (1 - t) + yk[:, np.newaxis] * t
        mid_th = th0[:, np.newaxis] * (1 - t) + thk[:, np.newaxis] * t

        mid_xy = tf.stack([mid_x, mid_y], axis=-1)
        dx = tf.linalg.norm(mid_xy[:, 0], axis=-1) * 1.0
        dxy = tf.stack([dx, tf.ones_like(dx)], axis=-1)[:, tf.newaxis]
        pts = pts * dxy
        r = Rot(mid_th)
        a = (r @ pts[..., tf.newaxis])[..., 0]
        p = a + mid_xy

        cp = tf.concat([xy0, xy1, xy2, p, xykm1, xyk], axis=1)
        return cp

class Loss:
    def __init__(self, n_pts):
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

        T = np.linspace(0., 1., 512)
        Ns = [np.stack([N(self.n, t, i) for i in range(self.m - self.n)]) for t in T]
        Ns = np.stack(Ns, axis=0)
        Ns[-1, -1] = 1.
        dNs = [np.stack([dN(self.n, t, i) for i in range(self.m - self.n)]) for t in T]
        dNs = np.stack(dNs, axis=0)
        dNs[-1, -1] = 21.
        dNs[-1, -2] = -21.
        ddNs = [np.stack([ddN(self.n, t, i) for i in range(self.m - self.n)]) for t in T]
        ddNs = np.stack(ddNs, axis=0)
        ddNs[-1, -1] = 294.
        ddNs[-1, -2] = -441.
        ddNs[-1, -3] = 147.
        return Ns[np.newaxis], dNs[np.newaxis], ddNs[np.newaxis]

    def auxiliary(self, plan, data):
        map, path, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)
        a0 = x0
        a1 = tf.ones_like(x0)
        a3 = -2 * (xk - x0 - 1.)
        a2 = 3 * (xk - x0 - 1.)
        b0 = y0
        b1 = tf.tan(th0)
        b3 = tf.tan(thk) + b1 - 2 * (yk - b0)
        b2 = yk - b1 - b0 - b3
        t = tf.linspace(0., 1., self.m - self.n)[tf.newaxis]
        xs = a3[:, tf.newaxis] * t**3 + a2[:, tf.newaxis] * t**2 + a1[:, tf.newaxis] * t + a0[:, tf.newaxis]
        ys = b3[:, tf.newaxis] * t**3 + b2[:, tf.newaxis] * t**2 + b1[:, tf.newaxis] * t + b0[:, tf.newaxis]
        #xs0 = x0[:, tf.newaxis] * (1 - t) + xk[:, tf.newaxis] * t
        #ys0 = y0[:, tf.newaxis] * (1 - t) + yk[:, tf.newaxis] * t
        xys = tf.stack([xs / self.dim, ys / (self.dim / 2)], axis=-1)
        dev = tf.abs(xys - plan)
        loss = tf.reduce_sum(dev, axis=(1, 2))

        x_plan = plan[:, :, 0] * self.dim
        y_plan = plan[:, :, 1] * (self.dim / 2.)
        plan_g = tf.stack([x_plan, y_plan], axis=-1)
        xy = self.N @ plan_g
        dxy = self.dN @ plan_g
        ddxy = self.ddN @ plan_g
        curvature = (ddxy[..., 1] * dxy[..., 0] - ddxy[..., 0] * dxy[..., 1]) / tf.reduce_sum(tf.square(dxy), axis=-1)**(3. / 2)
        curvature_loss = tf.reduce_mean(tf.nn.relu(tf.abs(curvature) - Car.max_curvature), -1)
        x_global = xy[..., 0]
        y_global = xy[..., 1]
        th_global = tf.atan2(dxy[..., 1], dxy[..., 0])
        curvature_loss = tf.reduce_mean(tf.nn.relu(tf.abs(curvature) - Car.max_curvature), -1)
        invalid_loss, supervised_loss = invalidate(x_global, y_global, th_global, map, path)

        #loss = curvature_loss

        loss += 1e-5 * curvature_loss
        return loss, invalid_loss, curvature_loss, curvature_loss, x_global, y_global, th_global



    def __call__(self, plan, data):
        map, path, x0, y0, th0, beta0, xk, yk, thk, betak = unpack_data(data)
        x_plan = plan[:, :, 0] * self.dim
        y_plan = plan[:, :, 1] * (self.dim / 2.)
        plan_g = tf.stack([x_plan, y_plan], axis=-1)
        xy = self.N @ plan_g
        dxy = self.dN @ plan_g
        ddxy = self.ddN @ plan_g
        curvature = (ddxy[..., 1] * dxy[..., 0] - ddxy[..., 0] * dxy[..., 1]) / tf.reduce_sum(tf.square(dxy), axis=-1)**(3. / 2)
        tcurv = tf.reduce_sum(tf.abs(curvature[:, 1:] - curvature[:, :-1]), axis=-1)
        #plt.plot(curvature.numpy()[0])
        #plt.ylim(-5, 5)
        #plt.show()
        curvature_loss = tf.reduce_sum(tf.nn.relu(tf.abs(curvature) - Car.max_curvature), -1)
        x_global = xy[..., 0]
        y_global = xy[..., 1]
        th_global = tf.atan2(dxy[..., 1], dxy[..., 0])
        #for i in range(16):
        #    plt.subplot(1, 2, 1)
        #    plt.plot(x_global[i], y_global[i])
        #    plt.plot(path[i, :, 0], path[i, :, 1])
        #    plt.xlim(0., self.dim)
        #    plt.ylim(-self.dim / 2, self.dim / 2)
        #    plt.subplot(1, 2, 2)
        #    plt.plot(curvature[i])
        #    plt.show()
        invalid_loss, supervised_loss = invalidate(x_global, y_global, th_global, map, path)
        #loss = invalid_loss + curvature_loss
        coarse_loss = invalid_loss + 1e-2 * curvature_loss
        fine_loss = invalid_loss + 1e-2 * curvature_loss + 1e-3 * tcurv
        loss = tf.where(coarse_loss == 0, fine_loss, coarse_loss)
        #loss = coarse_loss
        return loss, invalid_loss, curvature_loss, curvature_loss, x_global, y_global, th_global


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
    #plt.plot(25.6 * cps[0, :, 0],  12.8 * cps[0, :, 1], 'bx')
    plt.plot(cps_u,  cps_v, 'bx')
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

