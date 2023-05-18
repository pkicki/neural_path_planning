import numpy as np

from utils.collisions import xy_to_local_map, collision_with_map
from utils.constants import Car
from utils.crucial_points_np import calculate_car_crucial_points_numpy

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
    from ompl import geometric as og


def plan(map_img, x, y, th):
    def isStateValid(state):
        x = state.getX()#state[0]
        y = state.getY()#state[1]
        th = state.getYaw()#state[2]
        if np.abs(th) > np.pi:
            return False
        # perform collision checking or check if other constraints are
        # satisfied
        cp = np.array(calculate_car_crucial_points_numpy(x, y, th))
        cpx_map, cpy_map = xy_to_local_map(cp[:, 0], cp[:, 1])
        collision = collision_with_map(map_img, cpx_map, cpy_map)
        #print(collision)
        #print(x, y, th)
        #plt.imshow(map_img)
        #plt.plot(cpy_map, cpx_map)
        #plt.show()
        return not collision

    space = ob.DubinsStateSpace(Car.max_R)

    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-15.)
    bounds.setHigh(15.)
    space.setBounds(bounds)

    # define a simple setup class
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    # create a start state
    start = ob.State(space)
    start().setX(0.)
    start().setY(0.)
    start().setYaw(0.)

    # create a goal state
    goal = ob.State(space)
    goal().setX(x)
    goal().setY(y)
    goal().setYaw(th)

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.05)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    planner = og.BITstar(si)
    ss.setPlanner(planner)
    ss.getSpaceInformation().setStateValidityCheckingResolution(0.005)

    # attempt to solve the problem
    solved = ss.solve(5.0)
    if not solved:
        return None
    print(solved)
    path = ss.getSolutionPath()
    path.interpolate(128)
    #states = [[x[i] for i in range(3)] for x in path.getStates()]
    states = [[x.getX(), x.getY(), x.getYaw()] for x in path.getStates()]
    #valid = [isStateValid(s) for s in states]
    #print("XD")
    #print(valid)
    return states
