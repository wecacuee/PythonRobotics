"""

Move to specified pose

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai(@Atsushi_twi)

P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7

"""

import matplotlib.pyplot as plt
import numpy as np
from random import random
from functools import partial
from collections import namedtuple

# simulation parameters
Kp_rho = 9
Kp_alpha = 15
Kp_beta = -3
dt = 0.01

show_animation = True


PolarState = np.ndarray
CartesianState = np.ndarray


def polar2cartesian(x: PolarState, state_goal : CartesianState) -> CartesianState:
    rho, alpha, beta = x
    x_goal, y_goal, theta_goal = state_goal
    phi = angdiff(theta_goal, beta)
    x_diff = rho * np.cos(phi)
    y_diff = rho * np.sin(phi)
    theta = normalize_angle(alpha + phi)
    return np.array([x_goal - x_diff,
                     y_goal - y_diff,
                     theta])


def cartesian2polar(state: CartesianState, state_goal : CartesianState) -> PolarState:
    x, y, theta = state
    x_goal, y_goal, theta_goal = state_goal

    x_diff = x_goal - x
    y_diff = y_goal - y

    # reparameterization
    rho = np.hypot(x_diff, y_diff)
    phi = np.arctan2(y_diff, x_diff)
    alpha = angdiff(phi , theta)
    beta = angdiff(theta_goal , phi)
    return np.array((rho, alpha, beta))



class PolarDynamics:
    def f(self, x : PolarState):
        return np.zeros_like(x)

    def g(self, x : PolarState):
        rho, alpha, beta = x
        return (np.array([[-np.cos(alpha), 0],
                          [np.sin(alpha)/rho, -1],
                          [-np.sin(alpha)/rho, 0]])
                if (rho > 1e-2) else
                np.array([[-1, 0],
                          [1, -1],
                          [-1, 0]]))

class CartesianDynamics:
    def f(self, x : CartesianState):
        return np.zeros_like(x)

    def g(self, state: CartesianState):
        x, y, theta = state
        return np.array([[np.cos(theta), 0],
                         [np.sin(theta), 0],
                         [0, 1]])


def normalize_angle(theta):
    # Restrict alpha and beta (angle differences) to the range
    # [-pi, pi] to prevent unstable behavior e.g. difference going
    # from 0 rad to 2*pi rad with slight turn
    return (theta + np.pi) % (2 * np.pi) - np.pi

def angdiff(thetap, theta):
    return normalize_angle(thetap - theta)


def cosdist(thetap, theta):
    return 1 - np.cos(thetap - theta)


class ControllerCLF:
    """
    Aicardi, M., Casalino, G., Bicchi, A., & Balestrino, A. (1995). Closed loop steering of unicycle like vehicles via Lyapunov techniques. IEEE Robotics & Automation Magazine, 2(1), 27-35.
    """
    def __init__(self, # simulation parameters
                 Kp = np.array([9, 15, 5])/10.,
                 u_dim = 2,
                 dynamics = PolarDynamics()):
        self.Kp = np.asarray(Kp)
        self.u_dim = 2
        self.dynamics = dynamics

    def _clf_terms(self, x):
        rho, alpha, beta = x
        return np.array((0.5 * self.Kp[0] * rho ** 2,
                         self.Kp[1] * cosdist(alpha, 0),
                         self.Kp[2] * cosdist(alpha, beta)
        ))

    def _clf(self, x):
        return self._clf_terms(x).sum()

    def _grad_clf(self, x):
        rho, alpha, beta = x
        return np.array((self.Kp[0] * rho ,
                         self.Kp[1] * np.sin(alpha) + self.Kp[2] * np.sin(alpha - beta) ,
                         - self.Kp[2] * np.sin(alpha - beta)))

    def _clc(self, x, u):
        f, g = self.dynamics.f, self.dynamics.g
        gclf = self._grad_clf(x)
        print("x :", x)
        print("clf terms :", self._clf_terms(x))
        print("grad_x clf:", gclf)
        print("grad_u clf:", gclf @ g(x))
        return gclf @ (f(x) + g(x) @ u) + 10*self._clf(x)

    def _cost(self, x, u):
        import cvxpy as cp # pip install cvxpy
        return cp.sum_squares(u)

    def control(self, x, t):
        import cvxpy as cp # pip install cvxpy
        x = np.asarray(x)
        uvar = cp.Variable(self.u_dim)
        uvar.value = np.zeros(self.u_dim)
        relax = cp.Variable(1)
        obj = cp.Minimize(self._cost(x, uvar) + 10*self._clc(x, uvar))
        #constr = (self._clc(x, uvar) + relax <= 0)
        problem = cp.Problem(obj)#, [constr])
        problem.solve(solver='GUROBI')
        if problem.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            # print("Optimal value: %s" % problem.value)
            pass
        else:
            raise ValueError(problem.status)
        # for variable in problem.variables():
        #     print("Variable %s: value %s" % (variable.name(), variable.value))
        return uvar.value


class ControllerPID:
    def __init__(self, # simulation parameters
                 Kp_rho = 9,
                 Kp_alpha = 15,
                 Kp_beta = -3):
        self.Kp_rho = Kp_rho
        self.Kp_alpha = Kp_alpha
        self.Kp_beta = Kp_beta

    def control(self, x, t):
        rho, alpha, beta = x
        v = Kp_rho * rho
        w = Kp_alpha * alpha + Kp_beta * beta
        if alpha > np.pi / 2 or alpha < -np.pi / 2:
            v = -v
        return [v, w]


def move_to_pose(x_start, y_start, theta_start, x_goal, y_goal, theta_goal,
                 controller=ControllerCLF()):
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle

    Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards the goal
    Kp_beta*beta rotates the line so that it is parallel to the goal angle
    """
    x = x_start
    y = y_start
    theta = theta_start

    x_diff = x_goal - x
    y_diff = y_goal - y

    x_traj, y_traj = [], []

    rho = np.hypot(x_diff, y_diff)
    count = 0
    while rho > 0.001:
        x_traj.append(x)
        y_traj.append(y)

        state = cartesian2polar(np.array([x, y, theta]),
                                np.array([x_goal, y_goal, theta_goal]))
        rho, alpha, beta = state
        # control
        v, w = controller.control(state, t=count)

        # simulation
        theta = theta + w * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt

        # visualization
        if show_animation:  # pragma: no cover
            plt.cla()
            plt.arrow(x_start, y_start, np.cos(theta_start),
                      np.sin(theta_start), color='r', width=0.1)
            plt.arrow(x_goal, y_goal, np.cos(theta_goal),
                      np.sin(theta_goal), color='g', width=0.1)
            plot_vehicle(x, y, theta, x_traj, y_traj)
        count = count + 1


def plot_vehicle(x, y, theta, x_traj, y_traj):  # pragma: no cover
    # Corners of triangular vehicle when pointing to the right (0 radians)
    p1_i = np.array([0.5, 0, 1]).T
    p2_i = np.array([-0.5, 0.25, 1]).T
    p3_i = np.array([-0.5, -0.25, 1]).T

    T = transformation_matrix(x, y, theta)
    p1 = np.matmul(T, p1_i)
    p2 = np.matmul(T, p2_i)
    p3 = np.matmul(T, p3_i)

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-')
    plt.plot([p3[0], p1[0]], [p3[1], p1[1]], 'k-')

    plt.plot(x_traj, y_traj, 'b--')

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

    plt.xlim(0, 20)
    plt.ylim(0, 20)

    plt.pause(dt)


def transformation_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])


def main():
    move_to_pose_configured = partial(move_to_pose,
                                      controller=ControllerCLF(
                                          dynamics=PolarDynamics()
                                      ))
    # move_to_pose_configured = partial(move_to_pose,
    #                                   controller=ControllerPID())
    for i in range(5):
        x_start = 20 * random()
        y_start = 20 * random()
        theta_start = 2 * np.pi * random() - np.pi
        x_goal = 20 * random()
        y_goal = 20 * random()
        theta_goal = 2 * np.pi * random() - np.pi
        print("Initial x: %.2f m\nInitial y: %.2f m\nInitial theta: %.2f rad\n" %
              (x_start, y_start, theta_start))
        print("Goal x: %.2f m\nGoal y: %.2f m\nGoal theta: %.2f rad\n" %
              (x_goal, y_goal, theta_goal))
        move_to_pose_configured(x_start, y_start, theta_start, x_goal, y_goal,
                                theta_goal)


if __name__ == '__main__':
    main()
