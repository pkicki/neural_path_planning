import tensorflow as tf


def Rot(fi):
    c = tf.cos(fi)
    s = tf.sin(fi)
    L = tf.stack([c, s], -1)
    R = tf.stack([-s, c], -1)
    return tf.stack([L, R], -1)


def _calculate_length(x, y):
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    lengths = tf.sqrt(dx ** 2 + dy ** 2)
    length = tf.reduce_sum(lengths, -1)
    return length, lengths