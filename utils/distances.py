#!/usr/bin/python
import numpy as np
import tensorflow as tf


def dist2vert(v, q):
    v = tf.expand_dims(v, 1)
    q = tf.expand_dims(q, 2)
    d = euclid(v, q)
    d = tf.reduce_min(d, 2)
    return d


def euclid(a, b=None):
    if b is None:
        return tf.sqrt(tf.reduce_sum(a ** 2, -1))
    return tf.sqrt(tf.reduce_sum((a - b) ** 2, -1))




def point2edge(verts, query_points):
    query_points = query_points[:, :, :, tf.newaxis]
    first_point_coords = verts[:, tf.newaxis, tf.newaxis, :-1]
    second_point_coords = verts[:, tf.newaxis, tf.newaxis, 1:]
    edge_vector = second_point_coords - first_point_coords
    edge_vector = edge_vector
    query_points_in_v1 = query_points - first_point_coords
    p = tf.reduce_sum(edge_vector * query_points_in_v1, -1)
    t = tf.reduce_sum(edge_vector * edge_vector, -1)
    w = p / (t + 1e-8)
    w = tf.where(w <= 0, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    w = tf.where(w >= 1, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    p = edge_vector * tf.expand_dims(w, -1) \
        + first_point_coords  # calcualte point on the edge
    return p - query_points


def point2vert(verts, query_points):
    return verts[:, tf.newaxis, tf.newaxis] - query_points[:, :, :, tf.newaxis]

def path_dist_cp(gt_path, path):
    #dists = tf.linalg.norm(tf.ones_like(gt_path[:, tf.newaxis]) - path[:, :, tf.newaxis], axis=-1)
    #dists = tf.linalg.norm(gt_path[:, tf.newaxis] - path[:, :, tf.newaxis], axis=-1)
    #dists = tf.reduce_sum(dists, axis=-1)
    #dists = tf.reduce_sum(dists, axis=-1)

    #p2v = tf.linalg.norm(gt_path[:, tf.newaxis] - path[:, :, tf.newaxis], axis=-1)
    #dists = tf.reduce_sum(p2v, axis=-1)
    #dists = tf.reduce_sum(dists, axis=-1)

    gt_path_gp = gt_path[:, :, 0]
    path_gp = path[:, :, 0]
    p2v = tf.abs(gt_path_gp[:, tf.newaxis] - path_gp[:, :, tf.newaxis])
    #p2v = tf.square(gt_path_gp[:, tf.newaxis] - path_gp[:, :, tf.newaxis])
    p2v = tf.reduce_sum(p2v, axis=-1)
    #p2v = tf.sqrt(p2v)
    #p2v = tf.linalg.norm(gt_path_gp[:, tf.newaxis] - path_gp[:, :, tf.newaxis], axis=-1)
    ind = tf.argmin(p2v, axis=-1)
    gt_path_closest = tf.gather(gt_path, ind, axis=1, batch_dims=1)
    dists = tf.abs(gt_path_closest - path)
    #dists = tf.square(gt_path_closest - path)
    dists = tf.reduce_sum(dists, axis=-1)
    #dists = tf.sqrt(dists)
    #dists = tf.linalg.norm(gt_path_closest - path, axis=-1)
    dists = tf.reduce_mean(dists, -1)
    #dists = tf.reduce_sum(dists, -1)
    #dists = tf.reduce_mean(p2v, axis=(-2, -1))
    #a = tf.reduce_min(dists)
    #b = tf.reduce_max(dists)
    return dists


def path_dist_thk(gt_path, path):
    #dists = tf.linalg.norm(tf.ones_like(gt_path[:, tf.newaxis]) - path[:, :, tf.newaxis], axis=-1)
    #dists = tf.linalg.norm(gt_path[:, tf.newaxis] - path[:, :, tf.newaxis], axis=-1)
    #dists = tf.reduce_sum(dists, axis=-1)
    #dists = tf.reduce_sum(dists, axis=-1)

    #p2v = tf.linalg.norm(gt_path[:, tf.newaxis] - path[:, :, tf.newaxis], axis=-1)
    #dists = tf.reduce_sum(p2v, axis=-1)
    #dists = tf.reduce_sum(dists, axis=-1)

    gt_path_gp = gt_path[:, :, :2]
    path_gp = path[:, :, :2]
    p2v = tf.abs(gt_path_gp[:, tf.newaxis] - path_gp[:, :, tf.newaxis])
    #p2v = tf.square(gt_path_gp[:, tf.newaxis] - path_gp[:, :, tf.newaxis])
    p2v = tf.reduce_sum(p2v, axis=-1)
    #p2v = tf.sqrt(p2v)
    #p2v = tf.linalg.norm(gt_path_gp[:, tf.newaxis] - path_gp[:, :, tf.newaxis], axis=-1)
    ind = tf.argmin(p2v, axis=-1)
    gt_path_closest = tf.gather(gt_path, ind, axis=1, batch_dims=1)
    w = np.array([1., 1., 2., 5.])[np.newaxis, np.newaxis]
    dists = tf.abs(gt_path_closest - path) * w
    #dists = tf.square(gt_path_closest - path)
    dists = tf.reduce_sum(dists, axis=-1)
    #dists = tf.sqrt(dists)
    #dists = tf.linalg.norm(gt_path_closest - path, axis=-1)
    dists = tf.reduce_mean(dists, -1)
    #dists = tf.reduce_sum(dists, -1)
    #dists = tf.reduce_mean(p2v, axis=(-2, -1))
    #a = tf.reduce_min(dists)
    #b = tf.reduce_max(dists)
    return dists

def path_dist(path, query_points):
    """

    :param verts: (N, V, 4, 2)
    :param query_points: (N, S, P, 2)
    :return:
    """
    p2v = point2vert(path, query_points)
    p2v = tf.linalg.norm(p2v, axis=-1)
    p = tf.linalg.norm(path, axis=-1)
    p = tf.tile(p[:, tf.newaxis, tf.newaxis], (1, 128, 5, 1))
    p2v = tf.where(p != 0, p2v, 1e10 * tf.ones_like(p2v))
    p2v = tf.reduce_min(p2v, axis=-1)
    return p2v


def path_line_dist(verts, query_points):
    """

    :param verts: (N, V, 4, 2)
    :param query_points: (N, S, P, 2)
    :return:
    """
    p2e = point2edge(verts, query_points)
    p2e = tf.linalg.norm(p2e, axis=-1)
    p2e = tf.reduce_min(p2e, axis=-1)
    p2v = point2vert(verts, query_points)
    p2v = tf.linalg.norm(p2v, axis=-1)
    p2v = tf.reduce_min(p2v, axis=-1)
    dists = tf.where(p2v < p2e, p2v, p2e)

    p = tf.linalg.norm(verts, axis=-1)
    p = tf.tile(p[:, tf.newaxis, tf.newaxis], (1, 128, 5, 1))
    dists = tf.where(p != 0, dists, 1e10 * tf.ones_like(dists))
    return dists

def dist(verts, query_points):
    def cross_product(a, b):
        return a[:, :, :, :, :, 0] * b[:, :, :, :, :, 1] - b[:, :, :, :, :, 0] * a[:, :, :, :, :, 1]

    def point2edge(verts, query_points):
        first_point_coords = verts
        second_point_coords = tf.roll(verts, -1, -2)
        edge_vector = second_point_coords - first_point_coords
        edge_vector = edge_vector[:, tf.newaxis, tf.newaxis]
        query_points_in_v1 = query_points[:, :, :, tf.newaxis, tf.newaxis] - first_point_coords[:, tf.newaxis,
                                                                             tf.newaxis]
        p = tf.reduce_sum(edge_vector * query_points_in_v1, -1)
        cross = cross_product(edge_vector, query_points_in_v1)
        inside = tf.logical_or(tf.reduce_all(cross > 0, -1), tf.reduce_all(cross < 0, -1))
        inside = tf.reduce_any(inside, -1)
        t = tf.reduce_sum(edge_vector * edge_vector, -1)
        w = p / (t + 1e-8)
        w = tf.where(w <= 0, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
        w = tf.where(w >= 1, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
        p = edge_vector * tf.expand_dims(w, -1) \
            + first_point_coords[:, tf.newaxis, tf.newaxis]  # calcualte point on the edge
        return p - query_points[:, :, :, tf.newaxis, tf.newaxis], inside

    def point2vert(verts, query_points):
        return verts[:, tf.newaxis, tf.newaxis] - query_points[:, :, :, tf.newaxis, tf.newaxis]

    """
    :param verts: (N, V, 4, 2)
    :param query_points: (N, S, P, 2)
    :return:
    """
    p2e, inside = point2edge(verts, query_points)
    p2v = point2vert(verts, query_points)
    p2e = tf.reduce_sum(tf.abs(p2e), axis=-1)
    p2v = euclid(p2v)
    dists = tf.concat([p2e, p2v], -1)
    dists = tf.reduce_min(dists, (-2, -1))
    dists = tf.where(inside, tf.zeros_like(dists), dists)
    return dists

def if_inside(map, points):
    map = map.numpy()
    res = 0.2
    bs, n, _, _ = map.shape
    points = points.numpy()
    x = points[np.arange(bs), :, :, 0]
    y = points[np.arange(bs), :, :, 1]
    x /= res
    y /= res
    # workaroundnawyjezdzanie z mapy
    x = np.clip((120 - x).astype(np.int32), 0, 127)
    y = np.clip((-y + 64).astype(np.int32), 0, 127)
    inside = map[np.arange(bs)[:, np.newaxis, np.newaxis], x, y, 0]
    return inside

