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
    #ALG = ["SST", "BITSTAR"]
    TIME = 0.05
    #ALG = "SST"
    #ALG = "INFORRTSTAR"
    #ALG = "BITSTAR"
    #ALG = "ABITSTAR"
    #ALG = "AITSTAR"
    #ALG = "RRTSTAR"
    TYPE = "DUBINS"
    #lengths = {"SST": [], "BITSTAR": [], "OURS": [], "RRTSTAR": [], "ABITSTAR": [], "AITSTAR": []}
    lengths = {"BITSTAR": [], "OURS": [], "ABITSTAR": [], "AITSTAR": []}
    curvs = {"BITSTAR": [], "OURS": [], "ABITSTAR": [], "AITSTAR": []}
    #curvs = {"SST": [], "BITSTAR": [], "OURS": [], "RRTSTAR": [], "ABITSTAR": [], "AITSTAR": []}
    for fname in glob("../data/all_test/*.results"):
        with open(fname, 'r') as fh:
            lines = fh.read().split("\n")[:-1]
            for l in lines:
                larr = l.split()
                alg = larr[0]
                t = float(larr[2])
                if alg in lengths.keys():
                    if t == TIME or alg == "OURS":
                        length = float(larr[4])
                        curv = float(larr[3])
                        lengths[alg].append(length)
                        curvs[alg].append(curv)
    print("LENGTHS:")
    ref = lengths["OURS"]
    results = []
    ks = []
    for k, v in lengths.items():
        ks.append(k)
        l = []
        for i in range(len(ref)):
            if ref[i] != -1 and v[i] != -1:
                #l.append(v[i] / ref[i])
                #l.append((v[i] - ref[i]) / ref[i])
                l.append(v[i])
            else:
                l.append(-1)
        results.append(l)
    results = np.array(results)
    valid = np.all(results != -1, axis=0)
    results = results[:, valid]
    print(results.shape)
    m = np.mean(results, axis=-1)
    print(ks)
    print(m)
    print(np.std(results, axis=-1))

    print("CURVS:")
    ref = curvs["OURS"]
    results = []
    ks = []
    for k, v in curvs.items():
        ks.append(k)
        l = []
        for i in range(len(ref)):
            if ref[i] != -1 and v[i] != -1:
                #l.append(v[i] - ref[i])
                #l.append((v[i] - ref[i]) / ref[i])
                l.append(v[i])
            else:
                l.append(-1)
        results.append(l)
    results = np.array(results)
    valid = np.all(results != -1, axis=0)
    results = results[:, valid]
    print(results.shape)
    m = np.mean(results, axis=-1)
    print(ks)
    print(m)
    print(np.std(results, axis=-1))


    #print("CURVS:")
    #for k, v in curvs.items():
    #    print(k, np.mean(v))

