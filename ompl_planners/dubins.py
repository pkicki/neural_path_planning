#!/usr/bin/env python


# Author: Mark Moll

import os
import sys
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, tan
from functools import partial

from .constants import Car
from .crucial_points import calculate_car_crucial_points, calculate_car_contour
from .environment import Environment

try:
    from ompl import base as ob
    from ompl import control as oc
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import base as ob
    from ompl import control as oc


def transform_to_img(x, y, env, cast=False):
    x = 120 - (x / env.resolution)
    y = 64 - (y / env.resolution)
    if cast:
        return x.astype(np.int32), y.astype(np.int32)
    return x, y

def isStateValid(state, env):
    # perform collision checking or check if other constraints are
    # satisfied
    x = state[0][0]
    y = state[0][1]
    th = state[1].value
    crucial_points = calculate_car_crucial_points(x, y, th)
    cp = calculate_car_contour(crucial_points)

    cp_cliped = np.clip(cp, -35.6, 35.6)
    if np.any(cp_cliped != cp):
        return False
    x, y = transform_to_img(cp_cliped[:, 0], cp_cliped[:, 1], env, cast=True)
    x = np.clip(x, 0, 127)
    y = np.clip(y, 0, 127)
    if_collision = env.map[x, y]
    return not np.any(if_collision)


def plan(p0, pk, fname, time):
    # construct the state space we are planning in
    space = ob.DubinsStateSpace(1 / Car.max_curvature)
    se2_bounds = ob.RealVectorBounds(2)
    se2_bounds.setLow(0, 0.)
    se2_bounds.setHigh(0, 25.6)
    se2_bounds.setLow(1, 0.)
    se2_bounds.setHigh(1, 25.6)
    se2_bounds.setLow(2, -pi)
    se2_bounds.setHigh(2, pi)
    space.setBounds(se2_bounds)

    # define a simple setup class
    ss = og.SimpleSetup(space)

    env = Environment(fname)
    validator = lambda x: isStateValid(x, env)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(validator))
    #ss.setStateValidityChecker()

    # create a start state
    start = ob.State(space)
    start[0] = p0[0]
    start[1] = p0[1]
    start[2] = p0[2]

    # create a goal state
    goal = ob.State(space)
    goal[0] = pk[0]
    goal[1] = pk[1]
    goal[2] = pk[2]

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.2)
    #ss.setStartAndGoalStates(start, goal, 0.4)

    si = ss.getSpaceInformation()
    si.setStateValidityCheckingResolution(1. / 128)
    planner = og.RRTstar(si)
    planner = og.SST(si)
    #planner = og.BFMT(si)
    #planner = og.BITstar(si)
    #planner = og.InformedRRTstar(si)
    #planner = og.ABITstar(si)
    #planner = og.AITstar(si)
    #planner = og.BKPIECE1(si)
    #planner = og.TRRT(si)
    #planner = og.RRTConnect(si)
    ss.setPlanner(planner)
    # attempt to solve the problem
    solved = ss.solve(time)
    #solved = ss.solve(5.5)
    #solved = ss.solve(10.0)

    valid = int(ss.haveExactSolutionPath())
    total_dtheta = -1.
    length = -1.
    total_k = -1.
    if valid == 1:
        path = ss.getSolutionPath()
        length = path.length()
        path.interpolate(6 * 128)
        path = np.array([[f[0][0], f[0][1], f[1].value] for f in path.getStates()])
        theta = path[:, 2]
        dtheta = np.abs(np.diff(theta))
        x = path[:, 0]
        y = path[:, 1]
        dist = np.sqrt(x**2 + y**2)
        dist = np.abs(np.diff(dist))
        k = dtheta / dist
        total_k = np.mean(k)
        #total_dtheta = np.sum(dtheta)
        #print()
        #print()
        #print("XD")
        #print()
        #print()
        #plt.imshow(env.map)
        #cp = calculate_car_crucial_points(path[:, 0], path[:, 1], path[:, 2])
        #for i in range(5):
        #    x, y = transform_to_img(cp[:, i, 0], cp[:, i, 1], env)
        #    plt.plot(y, x)
        #x, y = transform_to_img(np.array(p0[0]), np.array(p0[1]), env)
        #plt.plot(y, x, 'gx')
        #x, y = transform_to_img(np.array(pk[0]), np.array(pk[1]), env)
        #plt.plot(y, x, 'rx')
        #plt.show()
    time_limit = time
    with open(fname.replace(".png", ".cba"), 'a') as fh:
        fh.write(" ".join(["SST", "DUBINS", str(time_limit), str(total_dtheta), str(length), str(valid)]) + "\n")
        #fh.write(" ".join(["INFORRTSTAR", "DUBINS", str(time_limit), str(total_dtheta), str(length), str(valid)]) + "\n")
        #fh.write(" ".join(["ABITSTAR", "DUBINS", str(time_limit), str(total_dtheta), str(length), str(valid)]) + "\n")
        #fh.write(" ".join(["AITSTAR", "DUBINS", str(time_limit), str(total_dtheta), str(length), str(valid)]) + "\n")
        #fh.write(" ".join(["RRTSTAR", "DUBINS", str(time_limit), str(total_dtheta), str(length), str(valid)]) + "\n")
        #fh.write(" ".join(["BITSTAR", "DUBINS", str(time_limit), str(total_dtheta), str(length), str(valid)]) + "\n")


#if solved:
##if ss.haveExactSolutionPath():
    #    # print the path to screen
    #    print("Validity solution:\n%s" % ss.haveExactSolutionPath())
    #    print("Found solution:\n%s" % ss.getSolutionPath().printAsMatrix())
    #    print("Time solution:\n%s" % ss.getLastPlanComputationTime())
    #    path = ss.getSolutionPath()
    #    path.interpolate(100)
    #    path = np.array([[f[0][0], f[0][1], f[1].value] for f in path.getStates()])
    #    #plt.imshow(env.map)
    #    # plt.xlim([0., 5.12])
    #    # plt.ylim([0., 5.12])
    #    theta = path[:, 2]
    #    dtheta = np.abs(np.diff(theta))
    #    total_dtheta = np.sum(dtheta)
    #    print("TOTAL DTHETA:", total_dtheta)
    #    plt.imshow(env.map)
    #    cp = calculate_car_crucial_points(path[:, 0], path[:, 1], path[:, 2])
    #    for i in range(5):
    #        x, y = transform_to_img(cp[:, i, 0], cp[:, i, 1], env)
    #        plt.plot(y, x)
    #    #plt.xlim([0., 25.6])
    #    #plt.ylim([0., 25.6])
    #    x, y = transform_to_img(np.array(p0[0]), np.array(p0[1]), env)
    #    plt.plot(y, x, 'gx')
    #    x, y = transform_to_img(np.array(pk[0]), np.array(pk[1]), env)
    #    plt.plot(y, x, 'rx')
    #    #for i in range(len(path)):
    #    #    plt.plot(path[i][0], path[i][1])
    #    #    plt.draw()
    #    #    plt.pause(0.05)
    #    plt.show()
    #    #plt.savefig('out.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    TIME = 3.0
    for fname in glob("../data/all_test/*.png"):
        print(fname)
        scn = fname.replace(".png", ".path")
        with open(scn, 'r') as fh:
            lines = fh.read().split("\n")[:-1]
            for l in lines:
                xythk = np.array(l.split()).astype(np.float32)
                ddy0 = xythk[0]
                xythk = np.reshape(xythk[1:], (-1, 4))
                p0 = xythk[0, :3].tolist()
                pk = xythk[-1, :3].tolist()
                plan(p0, pk, fname, TIME)
