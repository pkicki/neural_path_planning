import tensorflow as tf


def params(q0, q1):
    x0, y0, dy0, ddy0 = q0
    x1, y1, dy1, ddy1 = q1

    ones = tf.ones_like(x0, dtype=tf.float32)
    zeros = tf.zeros_like(x0, dtype=tf.float32)

    X0 = tf.stack([x0 ** 5, x0 ** 4, x0 ** 3, x0 ** 2, x0, ones], -1)
    dX0 = tf.stack([5 * x0 ** 4, 4 * x0 ** 3, 3 * x0 ** 2, 2 * x0, ones, zeros], -1)
    ddX0 = tf.stack([20 * x0 ** 3, 12 * x0 ** 2, 6 * x0, 2 * ones, zeros, zeros], -1)

    X1 = tf.stack([x1 ** 5, x1 ** 4, x1 ** 3, x1 ** 2, x1, ones], -1)
    dX1 = tf.stack([5 * x1 ** 4, 4 * x1 ** 3, 3 * x1 ** 2, 2 * x1, ones, zeros], -1)
    ddX1 = tf.stack([20 * x1 ** 3, 12 * x1 ** 2, 6 * x1, 2 * ones, zeros, zeros], -1)

    A = tf.stack([X0, dX0, ddX0, X1, dX1, ddX1], -2)
    b = tf.stack([y0, dy0, ddy0, y1, dy1, ddy1], -1)
    #print(A)
    h = tf.linalg.solve(A, b[..., tf.newaxis])
    return h


def curvature(p, x):
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    ddX = tf.stack([20 * x ** 3, 12 * x ** 2, 6 * x, 2 * ones, zeros, zeros], -1)
    dX = tf.stack([5 * x ** 4, 4 * x ** 3, 3 * x ** 2, 2 * x, ones, zeros], -1)
    ddY = tf.squeeze(ddX @ p, -1)
    dY = tf.squeeze(dX @ p, -1)
    curv = ddY / (1 + dY ** 2) ** (3 / 2)
    return curv, dX, dY

def DY(p, x):
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    dX = tf.stack([5 * x ** 4, 4 * x ** 3, 3 * x ** 2, 2 * x, ones, zeros], -1)
    dY = tf.squeeze(dX @ p, -1)
    return dY
