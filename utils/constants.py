import numpy as np


# class Car:
#    ## Kia Rio III hatchback dimensions
#    L = 2.5
#    rear_axle_to_front = 3.375
#    rear_axle_to_back = 0.67
#    W = 1.72
#    max_curvature = 1 / (5.3 - 1.792 / 2)
#
class Car:
    ## F1/10 bolid
    L = 0.324
    rear_axle_to_front = L + 0.114
    rear_axle_to_back = 0.11
    W = 0.3
    max_steering_angle = 0.6108652382
    max_R = L / np.tan(max_steering_angle)
    max_curvature = 1 / max_R


class Map:
    scale = 0.05  # m/px
    origin = [-30.5, -12.9, 0]
    W = 128
    H = 256
