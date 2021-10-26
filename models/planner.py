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

    def __init__(self, activation=tf.keras.activations.tanh, kernel_init_std=0.1, bias=0.0, mul=1., pre_mul=1.,
                 pre_bias=0.0):
        super(EstimatorLayer, self).__init__()
        self.bias = bias
        self.mul = mul
        self.pre_mul = pre_mul
        self.pre_bias = pre_bias
        self.activation = activation
        self.features = [
            tf.keras.layers.Dense(128, tf.nn.tanh,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(64, tf.nn.tanh,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(64, tf.nn.tanh,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
        ]
        self.out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std))

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.features:
            x = layer(x)
        x = self.out(x)
        x *= self.pre_mul
        x += self.pre_bias
        x = self.activation(x)
        x *= self.mul
        x += self.bias
        return x


class FeatureExtractorLayer(tf.keras.Model):
    """
    Feature exrtactor layer
    """

    def __init__(self, num_features, input_shape, activation=tf.keras.activations.tanh, kernel_init_std=0.1):
        super(FeatureExtractorLayer, self).__init__()
        self.features = [
            tf.keras.layers.Dense(64, activation),
                                  #kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(num_features, activation),
                                  #kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(num_features, activation),
                                  #kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(num_features, activation),
                                  #kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            # tf.keras.layers.Dense(num_features, activation),
        ]
        # self.fc = tf.keras.layers.Dense(num_features, activation)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.features:
            x = layer(x)
        # x = self.fc(x)
        return x


class MapFeaturesProcessor(tf.keras.Model):
    def __init__(self, num_features):
        super(MapFeaturesProcessor, self).__init__()
        self.num_features = 32
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
            tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh),
        ]

    def call(self, inputs, training=None):
        x = inputs
        bs = x.shape[0]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, -1))
        for layer in self.fc:
            x = layer(x)
        return x

def unpack_data(data):
    map, path, task_map, ddy0 = data
    path = path[..., :3]
    p0 = path[:, 0]
    x0, y0, th0 = tf.unstack(p0, axis=-1)
    pk = path[:, -1]
    xk, yk, thk = tf.unstack(pk, axis=-1)
    return map, task_map, path, x0, y0, th0, ddy0, xk, yk, thk


class PlanningNetworkMP(tf.keras.Model):

    def __init__(self, num_segments, input_shape):
        super(PlanningNetworkMP, self).__init__()

        n = 256
        self.num_segments = num_segments - 1

        self.map_processing = MapFeaturesProcessor(64)

        self.preprocessing_stage = FeatureExtractorLayer(n, input_shape)
        self.x_est = EstimatorLayer(tf.nn.sigmoid, mul=10., bias=0.1, kernel_init_std=0.1, pre_bias=-1., pre_mul=1.0)
        self.y_est = EstimatorLayer(tf.identity)
        self.dy_est = EstimatorLayer(tf.identity)
        self.ddy_est = EstimatorLayer(tf.identity)
        #self.last_ddy_est = EstimatorLayer(tf.identity)

    def call(self, data, map_features, training=None):
        map, task_map, path, x0, y0, th0, ddy0, xk, yk, thk = unpack_data(data)
        last_ddy = ddy0
        W = 25.6
        H = 25.6

        map = tf.concat([map, task_map], axis=-1)
        map_features = self.map_processing(map)

        parameters = []
        for i in range(self.num_segments):
            inputs = tf.stack([x0 / W, y0 / H, np.sin(th0), np.cos(th0), last_ddy,
                               xk / W, yk / H, np.sin(thk), np.cos(thk)], -1)

            features = self.preprocessing_stage(inputs, training)
            features = tf.concat([features, map_features], -1)

            x = self.x_est(features, training)
            y = self.y_est(features, training)
            dy = self.dy_est(features, training)
            ddy = self.ddy_est(features, training)
            p = tf.concat([x, y, dy, ddy], -1)
            parameters.append(p)

            x0, y0, th0 = calculate_next_point(p, x0, y0, th0, last_ddy)
            last_ddy = ddy[:, 0]

        #last_ddy = self.last_ddy_est(features)
        parameters = tf.stack(parameters, -1)
        last_ddy = None

        return parameters, last_ddy


def calculate_next_point(plan, xL, yL, thL, last_ddy):
    x = plan[:, 0]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

    return x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


def plan_loss(plan, data, very_last_ddy):
    num_gpts = plan.shape[-1]
    map, task_map, path, x0, y0, th0, ddy0, xk, yk, thk = unpack_data(data)
    xL = x0
    yL = y0
    thL = th0
    last_ddy = ddy0
    curvature_loss = 0.0
    obstacles_loss = 0.0
    length_loss = 0.0
    curvature_accumulation_loss = 0.0
    supervised_loss = 0.0
    lengths = []
    x_path = []
    y_path = []
    th_path = []
    # regular path
    for i in range(num_gpts):
        x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL, curvature_sum, super_loss = \
            process_segment(plan[:, :, i], xL, yL, thL, last_ddy, map, path)
        curvature_loss += tf.nn.relu(curvature_violation)
        #print("CV", i, curvature_violation)
        #print("INV", i, invalid)
        obstacles_loss += tf.nn.relu(invalid)
        curvature_accumulation_loss += curvature_sum
        supervised_loss += super_loss

        length_loss += length
        lengths.append(length)
        x_path.append(x_glob)
        y_path.append(y_glob)
        th_path.append(th_glob)
        dth = tf.atan(plan[:, 2, i])
        m = tf.cos(dth)**3
        last_ddy = plan[:, 3, i] * m

    lengths = tf.stack(lengths, -1)

    dx = tf.nn.relu(tf.abs(xk - xL) - 0.2)
    dy = tf.nn.relu(tf.abs(yk - yL) - 0.2)
    dth = 10 * tf.nn.relu(tf.abs(thk - thL) - 0.05)
    overshoot_loss = dx + dy + dth

    # loss for training
    curvature_loss *= 1e1
    coarse_loss = tf.nn.relu(curvature_loss) + tf.nn.relu(obstacles_loss) + tf.nn.relu(overshoot_loss)
    fine_loss = tf.nn.relu(curvature_loss) + tf.nn.relu(obstacles_loss) + tf.nn.relu(overshoot_loss) + 1e-4 * tf.nn.relu(curvature_accumulation_loss)
    loss = tf.where(curvature_loss + obstacles_loss + overshoot_loss == 0, fine_loss, coarse_loss)
    #loss = coarse_loss

    return loss, obstacles_loss, overshoot_loss, curvature_loss, curvature_accumulation_loss, supervised_loss, x_path, y_path, th_path


def _plot(x_path, y_path, th_path, data, step, print=False):
    res = 0.2
    free_space, path, _, ddy0 = data
    path = path[..., :3]
    plt.imshow(free_space[0,..., 0])
    for i in range(len(x_path)):
        x = x_path[i][0]
        y = y_path[i][0]
        th = th_path[i][0]
        cp = calculate_car_crucial_points(x, y, th)
        for p in cp:
            u = -p[:, 1] / res + 64
            v = 120 - p[:, 0] / res
            plt.plot(u, v)
            #plt.plot(p[:, 0] * 10, (10. - p[:, 1]) * 10)
    #plt.plot(path[0, :, 0] * 10, (10 - path[0, :, 1]) * 10, 'r')
    u = -path[0, :, 1] / res + 64
    v = 120 - path[0, :, 0] / res
    plt.plot(u, v, 'r')
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


def process_segment(plan, xL, yL, thL, last_ddy, free_space, path):
    x = plan[:, 0]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

    # calcualte length of segment
    length, segments = _calculate_length(x_glob, y_glob)

    # calculate violations
    curvature_violation = tf.reduce_sum(tf.nn.relu(tf.abs(curvature[:, 1:]) - Car.max_curvature) * segments, -1)
    curvature_sum = tf.reduce_sum(tf.abs(curvature), -1)
    invalid, supervised_loss = invalidate(x_glob, y_glob, th_glob, free_space, path)

    return x_glob, y_glob, th_glob, curvature_violation, invalid, length, x_glob[:, -1], y_glob[:, -1], th_glob[:, -1], curvature_sum, supervised_loss


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


def _calculate_global_xyth_and_curvature(params, x, xL, yL, thL):
    x_local_sequence = tf.expand_dims(x, -1)
    x_local_sequence *= tf.linspace(0.0, 1.0, 64)
    #x_local_sequence *= tf.linspace(0.0, 1.0, 128)
    curv, dX, dY = curvature(params, x_local_sequence)
    x_glob, y_glob, th_glob = _calculate_global_xyth(params, x_local_sequence, xL, yL, thL, dY)
    return x_glob, y_glob, th_glob, curv


def _calculate_global_xyth(params, x, xL, yL, thL, dY):
    X = tf.stack([x ** 5, x ** 4, x ** 3, x ** 2, x, tf.ones_like(x)], -1)
    y = tf.squeeze(X @ params, -1)
    R = Rot(thL)
    xy_glob = R @ tf.stack([x, y], 1)
    xyL = tf.stack([xL, yL], -1)[..., tf.newaxis]
    xy_glob += tf.constant(xyL, dtype=tf.float32)
    x_glob, y_glob = tf.unstack(xy_glob, axis=1)
    th_glob = thL[:, tf.newaxis] + tf.atan(dY)
    return x_glob, y_glob, th_glob
