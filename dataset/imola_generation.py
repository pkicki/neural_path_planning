import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2

CURRENT_FILE = os.path.realpath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from utils.collisions import xy_to_local_map, collision_with_map
from utils.constants import Car, Map
from utils.crucial_points_np import calculate_car_crucial_points_numpy
from utils.ompl_planning import plan

#N = 10000
scenarios_per_map = 5

track_map = cv2.imread("../data/imola/imola.png", cv2.IMREAD_GRAYSCALE)
print(track_map.shape)

local_car_image = np.zeros((Map.H, Map.W))
car_H = int(Car.rear_axle_to_front / Map.scale)
car_W = Car.W / (2 * Map.scale)
local_car_image[Map.H - car_H:, int(Map.W/2 - car_W):int(Map.W/2 + car_W)] = 1.

def xytotrackimg(x, y):
    u = (x - Map.origin[0]) / Map.scale
    v = track_map.shape[0] - (y - Map.origin[1]) / Map.scale
    return u, v

def uvtoxy(u, v):
    x = u * Map.scale + Map.origin[0]
    y = (track_map.shape[0] - v) * Map.scale + Map.origin[1]
    return x, y


def validate_local_map(img):
    #car_H = int(Car.rear_axle_to_front / Map.scale)
    #car_W = Car.W / (2 * Map.scale)
    #car_img = img[Map.H - car_H:, int(Map.W / 2 - car_W):int(Map.W / 2 + car_W)]
    #if np.all(car_img > 0.):
    #    return True
    #return False

    #plt.subplot(131)
    #plt.imshow(img)
    #plt.subplot(132)
    #plt.imshow(local_car_image)
    #plt.subplot(133)
    #plt.show()
    #pass

    cp = np.array(calculate_car_crucial_points_numpy(0., 0., 0.))
    x_map, y_map = xy_to_local_map(cp[:, 0], cp[:, 1])
    collision = collision_with_map(img, x_map, y_map)
    return not collision



def subimage(image, center, theta, width, height):
    '''
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''
    print(center, theta)
    theta = -theta
    theta += np.pi / 2
    theta *= 180 / np.pi

    pad = np.maximum(Map.W, Map.H)
    center = (center[0] + pad, center[1] + pad)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = np.pad(image, [[pad, pad], [pad, pad]])
    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)
    #plt.subplot(221)
    #plt.imshow(image)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)
    #plt.subplot(222)
    #plt.imshow(image)

    #x = int(center[0] - width / 2)
    #y = int(center[1] - height / 2)
    x = int(center[0] - width / 2)
    y = int(center[1])# - height / 2)
    image = image[y - height:y, x:x + width]
    #plt.subplot(223)
    #plt.imshow(image)
    #plt.show()
    #image = image[::-1, ::-1]
    return image


map_traj_columns = ['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2']
raceline = np.loadtxt("../data/imola/track_2_narrow.csv", delimiter=";")
print(raceline.shape)

x = raceline[:, 1]
y = raceline[:, 2]
dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
th = raceline[:, 3] + np.pi/2

#plt.subplot(121)
#plt.imshow(track_map)
#plt.subplot(122)
#plt.plot(x, y)
#for i in range(0, raceline.shape[0], 100):
#    plt.arrow(x[i], y[i], np.cos(th[i]), np.sin(th[i]))
#plt.show()

u, v = xytotrackimg(x, y)

th_std = np.pi/20.
xy_std = 0.5

i = int(sys.argv[1])
N = int(sys.argv[2])
while i < N:
    idx = i % raceline.shape[0]
    ub, vb = xytotrackimg(x[idx], y[idx])
    xp = x[idx] + np.random.normal(scale=xy_std)
    yp = y[idx] + np.random.normal(scale=xy_std)
    dthp = np.random.normal(scale=th_std)
    thp = th[idx] + dthp
    print("TH: ", th[idx], dthp)

    up, vp = xytotrackimg(xp, yp)
    r = Map.H / 2
    uc = up + np.cos(thp) * r
    vc = vp - np.sin(thp) * r
    #cropped = subimage(track_map, (uc, vc), thp, Map.W, Map.H)
    cropped = subimage(track_map, (up, vp), thp, Map.W, Map.H)

    is_valid = validate_local_map(cropped)
    print(is_valid)
    #plt.subplot(121)
    #plt.imshow(track_map)
    #plt.plot([up], [vp], 'bx')
    #plt.plot([uc], [vc], 'rx')
    #plt.plot([ub], [vb], 'gx')
    #plt.plot(u, v)
    #plt.subplot(122)
    #plt.imshow(cropped)
    #plt.plot([Map.W / 2], [Map.H / 2], 'rx')
    #plt.show()

    if not is_valid:
        continue

    cv2.imwrite(f"../data/imola/cropped_maps/{i:06d}.png", cropped)

    k = 0
    while k < scenarios_per_map:
        x_offset = 5.
        expected_dist = np.clip(np.random.normal(loc=9., scale=3.), x_offset, Map.H * Map.scale)
        print(expected_dist)
        goal_idx = (idx + np.around(expected_dist / 0.05).astype(np.int32)) % raceline.shape[0]
        print("START IDX:", idx, " GOAL IDX:", goal_idx)
        goal_x = x[goal_idx] + np.random.normal(scale=xy_std)
        goal_y = y[goal_idx] + np.random.normal(scale=xy_std)
        goal_th = th[goal_idx] + np.random.normal(scale=th_std)

        goal_th = np.arctan2(np.sin(goal_th), np.cos(goal_th))
        thp = np.arctan2(np.sin(thp), np.cos(thp))
        goal_th_in_p = goal_th - thp
        goal_th_in_p = np.arctan2(np.sin(goal_th_in_p), np.cos(goal_th_in_p))
        dx = goal_x - xp
        dy = goal_y - yp
        goal_x_in_p = np.cos(thp) * dx + np.sin(thp) * dy
        goal_y_in_p = np.cos(thp) * dy - np.sin(thp) * dx

        if goal_x_in_p > Map.H * Map.scale:
            continue

        print("START:")
        print(xp, yp, thp)
        print("END:")
        print(goal_x, goal_y, goal_th)
        print("END IN START:")
        print(goal_x_in_p, goal_y_in_p, goal_th_in_p)
        ug, vg = xy_to_local_map(goal_x_in_p, goal_y_in_p)
        print("END UV:", ug, vg)

        cp = np.array(calculate_car_crucial_points_numpy(goal_x_in_p, goal_y_in_p, goal_th_in_p))
        cpx_map, cpy_map = xy_to_local_map(cp[:, 0], cp[:, 1])
        collision = collision_with_map(cropped, cpx_map, cpy_map)
        print("COLISION:", collision)

        #plt.subplot(121)
        #plt.imshow(track_map)
        #plt.plot([up], [vp], 'bx')
        #plt.plot([uc], [vc], 'rx')
        #plt.plot([ub], [vb], 'gx')
        #plt.plot(u, v)
        #plt.subplot(122)
        #plt.imshow(cropped)
        #plt.plot([vg], [ug], 'gx')
        #plt.plot(cpy_map, cpx_map)
        #plt.plot([Map.W / 2], [Map.H / 2], 'rx')

        #plt.show()

        if collision:
            print("END POSE INVALID")
            continue

        start_beta = Car.max_steering_angle * (2 * np.random.random() - 1.)
        path = plan(cropped, goal_x_in_p, goal_y_in_p, goal_th_in_p)

        path_valid = path is not None
        if path is None:
            print("PATH INVALID")
            continue

        path = np.array(path)
        px_map, py_map = xy_to_local_map(path[:, 0], path[:, 1])

        #plt.subplot(121)
        #plt.imshow(cropped)
        #plt.plot(ug, vg)
        #plt.plot(py_map, px_map)
        #plt.plot(cpy_map, cpx_map)
        #plt.subplot(122)
        #plt.plot(cp[:, 0], cp[:, 1])
        #plt.plot(path[:, 0], path[:, 1])
        #plt.show()

        np.savetxt(f"../data/imola/paths/{i:06d}_{k}.path", path, delimiter="\t")
        k += 1
    cv2.imwrite(f"../data/imola/cropped_maps/{i:06d}.png", cropped)
    i += 1




#u, v = xytotrackimg(x, y)
#th = -th
#
#r = Map.H / 2
##uc = u - np.sin(th) * r
##vc = v + np.cos(th) * r
#uc = u + np.sin(th) * r
#vc = v - np.cos(th) * r
#
#i = 0
##i = 4325
#while i < N:
#    idx = i % raceline.shape[0]
#    u_noise = int(1.5 / Map.scale * (2 * np.random.rand() - 1))
#    v_noise = int(1.5 / Map.scale * (2 * np.random.rand() - 1))
#    #th_noise = np.pi / 6 * (2 * np.random.rand() - 1)
#    th_noise = np.random.normal(scale=np.pi/6)
#    #u_noise = 0.
#    #v_noise = 0.
#    #th_noise = 0.
#    new_u = uc[idx] + u_noise
#    new_v = vc[idx] + v_noise
#    new_th = th[idx] + th_noise
#    cropped = subimage(track_map, (new_u, new_v), new_th, Map.W, Map.H)
#    #cropped = cropped[::-1]
#
#    is_valid = validate_local_map(cropped)
#    print(is_valid)
#
#
#    #plt.subplot(121)
#    #plt.imshow(track_map)
#    #plt.plot([u[idx]], [v[idx]], 'bx')
#    #plt.plot([uc[idx]], [vc[idx]], 'rx')
#    #plt.plot([uc[idx] + u_noise], [vc[idx] + v_noise], 'gx')
#    #plt.plot(u, v)
#    #plt.subplot(122)
#    #plt.imshow(cropped)
#    #plt.show()
#    if not is_valid:
#        continue
#
#    # plan path
#    start_x = x[idx]
#    start_y = y[idx]
#    start_th = th[idx]
#
#    start_x, start_y = uvtoxy(new_u, new_v)
#    x_offset = 5.
#    expected_dist = np.clip(np.random.normal(loc=10., scale=3.), x_offset, Map.H * Map.scale)
#    goal_idx = idx + np.around(expected_dist / 0.05).astype(np.int32)
#    goal_x = x[goal_idx] + np.random.normal(scale=2.)
#    goal_y = y[goal_idx] + np.random.normal(scale=2.)
#    goal_th = th[goal_idx] + np.random.normal(scale=np.pi / 6)
#    # check if goal in map
#    goal_th_in_start = goal_th - start_th
#    goal_x_in_start = ...
#
#    cv2.imwrite(f"../data/imola/cropped_maps/{i:06d}.png", cropped)
#    i += 1
