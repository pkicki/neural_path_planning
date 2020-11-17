#!/usr/bin/env python


# Author: Mark Moll

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, tan
from functools import partial

from constants import Car
from crucial_points import calculate_car_crucial_points, calculate_car_contour
from environment import Environment

if __name__ == "__main__":
    TIME = 10.0
    #ALG = "SST"
    #ALG = "INFORRTSTAR"
    #ALG = "BITSTAR"
    #ALG = "ABITSTAR"
    #ALG = "AITSTAR"
    ALG = "RRTSTAR"
    #ALG = "OURS"
    TYPE = "DUBINS"
    #TYPE = "SPLINE"
    acc = []
    for fname in glob("../data/all_test/*.results"):
        with open(fname, 'r') as fh:
            lines = fh.read().split("\n")[:-1]
            for l in lines:
                if l.startswith(ALG + " " + TYPE + " " + str(TIME)):
                    data = l.split()
                    valid = float(data[-1])
                    acc.append(valid)
    print("ACCURACY:", np.mean(acc))
