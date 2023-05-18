import numpy as np
from utils.constants import Map


def xy_to_local_map(x, y):
    x_map = Map.H - x / Map.scale
    y_map = Map.W / 2 - y / Map.scale
    return x_map, y_map

def collision_with_map(map_img, x, y):
    #x_idx = np.around(x).astype(np.int32)
    #y_idx = np.around(y).astype(np.int32)
    #if np.any(x_idx < 0) or np.any(x_idx > Map.H - 1) or np.any(y_idx < 0) or np.any(y_idx > Map.W - 1):
    #    return False
    x_idx = np.clip(np.around(x).astype(np.int32), 0, Map.H - 1)
    y_idx = np.clip(np.around(y).astype(np.int32), 0, Map.W - 1)
    return np.any(map_img[x_idx, y_idx] < 127)
