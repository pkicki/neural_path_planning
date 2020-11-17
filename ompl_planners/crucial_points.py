import numpy as np
from constants import Car


def calculate_car_crucial_points(x, y, fi):
    pose = np.stack([x, y], -1)
    cfi = np.cos(fi)
    sfi = np.sin(fi)
    cs = np.stack([cfi, sfi], -1)
    msc = np.stack([-sfi, cfi], -1)
    front_center = pose + Car.rear_axle_to_front * cs
    back_center = pose - Car.rear_axle_to_back * cs
    front_left = front_center + msc * Car.W / 2
    front_right = front_center - msc * Car.W / 2
    back_left = back_center + msc * Car.W / 2
    back_right = back_center - msc * Car.W / 2
    return np.stack([pose, front_left, front_right, back_left, back_right], axis=-2)


def calculate_car_contour(cp):
    def connect(a, b, norm):
        s = np.linspace(0., 1., int(norm / 0.2))[:, np.newaxis]
        return s * a[np.newaxis] + (1 - s) * b[np.newaxis]

    _, a, b, c, d = cp
    w = Car.W
    l = Car.rear_axle_to_front + Car.rear_axle_to_back
    ab = connect(a, b, w)
    bd = connect(b, d, l)
    dc = connect(d, c, w)
    ca = connect(c, a, l)
    contour = np.concatenate([ab, bd, dc, ca], axis=0)
    return contour
