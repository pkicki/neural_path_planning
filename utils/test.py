from models.planner import PlanningNetworkMP, Loss
from utils.crucial_points import calculate_car_crucial_points
import tensorflow as tf
import numpy as np

from utils.execution import ExperimentHandler


def read_map(map_path):
    img = tf.io.read_file(map_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    free = img > 0.5
    obs = img < 0.5
    img = tf.cast(tf.concat([free, obs], axis=-1), tf.float32)
    return img


def transform_to_img(x, y):
    x = 120 - (x / .2)
    y = 64 - (y / .2)
    return y, x


def _plot_car(cp, ax, c='m'):
    x = [p[0].numpy() for p in cp[1:]]
    y = [p[1].numpy() for p in cp[1:]]
    tmp = x[2]
    x[2] = x[3]
    x[3] = tmp
    tmp = y[2]
    y[2] = y[3]
    y[3] = tmp
    x = np.array(x + [x[0]])
    y = np.array(y + [y[0]])
    x, y = transform_to_img(x, y)
    ax.fill(x, y, c, alpha=0.5, zorder=3)


def _plot(x_path, y_path, th_path, ax):
    x_path = tf.concat(x_path, 0)
    y_path = tf.concat(y_path, 0)
    th_path = tf.concat(th_path, 0)
    path = tf.stack([x_path, y_path, th_path], axis=-1)
    cp = calculate_car_crucial_points(x_path, y_path, th_path)
    cp = tf.stack(cp, axis=1)
    _plot_car(cp[0, :, 0], ax, 'g')
    cl = ['c', 'g', 'b', 'm', 'k']
    for s in range(path.shape[0]):
        x = x_path[s]
        y = y_path[s]
        th = th_path[s]
        cps = calculate_car_crucial_points(x, y, th)
        for j, p in enumerate(cps):
            x, y = transform_to_img(p[:, 0], p[:, 1])
            ax.plot(x, y, color=cl[j], zorder=4)
    _plot_car(cp[-1, :, -1], ax, 'r')


def run_and_plot(model_path, map_path, as_path, xd, yd, thd, ax):
    bs = 128
    map = read_map(map_path)[tf.newaxis]
    p0 = np.array([0.4, 0., 0., 0.], dtype=np.float32)[np.newaxis]
    pk = np.array([xd, yd, thd, 0.], dtype=np.float32)[np.newaxis]
    path = np.stack([p0, pk], axis=1)
    ddy0 = np.array([0.], dtype=np.float32)
    #data = (map, path, ddy0)
    data = (map, path)

    # 2. Define model
    N = 3
    model = PlanningNetworkMP(N)
    loss = Loss(N)

    # 3. Optimization

    optimizer = tf.keras.optimizers.Adam(1e-4)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(".", "", 1, model, optimizer)
    experiment_handler.restore(model_path)

    output, _ = model(data, None, training=True)
    #model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, _, x_path, y_path, th_path = loss(
    #    output, data)
    model_loss, invalid_loss, curvature_loss, overshoot_loss, total_curvature_loss, x_path, y_path, th_path, curvature = loss(
        output, data)
    print(invalid_loss+curvature_loss, invalid_loss, curvature_loss)

    _plot(x_path, y_path, th_path, ax)

    ax.imshow(map[0, ..., 0], cmap='gray')
    if as_path is not None:
        uvc = np.loadtxt(as_path, delimiter="\t")
        u, v, c = np.split(uvc, 3, axis=-1)
        c = 181. * c
        return ax.scatter(u, v, c=c, s=1.5 * np.ones_like(c),
                          zorder=3, cmap='hot_r')
    return x_path, y_path, th_path