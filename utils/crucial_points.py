import tensorflow as tf
from utils.constants import Car
from utils.utils import Rot
import numpy as np


def calculate_car_crucial_points(x, y, fi):
    pose = tf.stack([x, y], -1)
    cfi = tf.cos(fi)
    sfi = tf.sin(fi)
    cs = tf.stack([cfi, sfi], -1)
    msc = tf.stack([-sfi, cfi], -1)
    front_center = pose + Car.rear_axle_to_front * cs
    back_center = pose - Car.rear_axle_to_back * cs
    front_left = front_center + msc * Car.W / 2
    front_right = front_center - msc * Car.W / 2
    back_left = back_center + msc * Car.W / 2
    back_right = back_center - msc * Car.W / 2
    return [pose, front_left, front_right, back_left, back_right]


def calculate_car_contour(cp):
    def connect(a, b, norm):
        s = tf.linspace(0., 1., int(norm / 0.2))[tf.newaxis, tf.newaxis, :, tf.newaxis]
        return s * a[:, :, tf.newaxis] + (1 - s) * b[:, :, tf.newaxis]

    _, a, b, c, d = cp
    w = Car.W
    l = Car.rear_axle_to_front + Car.rear_axle_to_back
    ab = connect(a, b, w)
    bd = connect(b, d, l)
    dc = connect(d, c, w)
    ca = connect(c, a, l)
    contour = tf.concat([ab, bd, dc, ca], axis=-2)
    return contour


def calculate_car_body(x, y, fi):
    X = tf.linspace(-Car.rear_axle_to_back, Car.rear_axle_to_front,
                    int((Car.rear_axle_to_back + Car.rear_axle_to_front) / 0.2))
    Y = tf.linspace(-Car.W / 2, Car.W / 2, int(Car.W / 0.2))
    X, Y = tf.meshgrid(X, Y)
    body = tf.stack([X, Y], axis=-1)
    body = body[tf.newaxis, tf.newaxis, :, :, :, tf.newaxis]
    pose = tf.stack([x, y], -1)[:, :, tf.newaxis, tf.newaxis]
    R = Rot(fi)[:, :, tf.newaxis, tf.newaxis]
    rotated_body = (R @ body)[..., 0]
    moved_body = rotated_body + pose
    return moved_body
