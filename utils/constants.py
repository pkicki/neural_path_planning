import numpy as np

class Car:
    ## Kia Rio III hatchback dimensions
    L = 2.5
    rear_axle_to_front = 3.375
    rear_axle_to_back = 0.67
    W = 1.72
    max_curvature = 1 / (5.3 - 1.792 / 2)

    front_left = np.array([rear_axle_to_front, W / 2], dtype=np.float32)[np.newaxis]
    front_right = np.array([rear_axle_to_front, -W / 2], dtype=np.float32)[np.newaxis]
    back_left = np.array([-rear_axle_to_back, W / 2], dtype=np.float32)[np.newaxis]
    back_right = np.array([-rear_axle_to_back, -W / 2], dtype=np.float32)[np.newaxis]

    crucial_points = np.concatenate([np.zeros_like(front_left), front_left, front_right, back_right, back_left], axis=0)

    tw = np.linspace(0., 1., int(W / 0.2))[:, np.newaxis]
    tl = np.linspace(0., 1., int((rear_axle_to_front + rear_axle_to_back) / 0.2))[:, np.newaxis]
    front = front_left * tw + (1 - tw) * front_right
    back = back_left * tw + (1 - tw) * back_right
    left = front_left * tl + (1 - tl) * back_left
    right = front_right * tl + (1 - tl) * back_right
    contour = np.concatenate([front, left, back, right], axis=0).astype(np.float32)
