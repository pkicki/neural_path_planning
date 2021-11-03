import tensorflow as tf


def transform_points(xy, R, points):
    return tf.einsum('...ij,...kj->...ik', points, R) + xy