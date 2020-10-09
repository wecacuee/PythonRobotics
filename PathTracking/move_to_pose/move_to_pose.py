"""

Move to specified pose

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai(@Atsushi_twi)

P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7

"""
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from random import random
from functools import partial
from collections import namedtuple
import sys
from torch.utils.tensorboard import SummaryWriter

# simulation parameters

PolarState = namedtuple('PolarState', 'rho alpha beta')
CartesianState = namedtuple('CartesianState', 'x y theta')

LOG = SummaryWriter('data/runs/' + datetime.now().strftime("%m%d-%H%M"))


def polar2cartesian(x: PolarState, state_goal : CartesianState) -> CartesianState:
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle

    >>> polar = np.random.rand(3) * np.array([1, 2*np.pi, 2*np.pi]) - np.array([0, np.pi, np.pi])
    >>> state_goal = np.random.rand(3) * np.array([2, 2, 2*np.pi]) - np.array([1, 1, np.pi])
    >>> state = polar2cartesian(polar, state_goal)
    >>> polarp = cartesian2polar(state, state_goal)
    >>> np.testing.assert_allclose(polar, polarp)
    """
    rho, alpha, beta = x
    x_goal, y_goal, theta_goal = state_goal
    phi = angdiff(theta_goal, beta)
    x_diff = rho * np.cos(phi)
    y_diff = rho * np.sin(phi)
    theta = angdiff(phi, alpha)
    return np.array([x_goal - x_diff,
                     y_goal - y_diff,
                     theta])


def cartesian2polar(state: CartesianState, state_goal : CartesianState) -> PolarState:
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle

    >>> state = np.random.rand(3)* np.array([2, 2, 2*np.pi]) - np.array([1, 1, np.pi])
    >>> state_goal = np.random.rand(3)* np.array([2, 2, 2*np.pi]) - np.array([1, 1, np.pi])
    >>> polar = cartesian2polar(state, state_goal)
    >>> statep = polar2cartesian(polar, state_goal)
    >>> np.testing.assert_allclose(state, statep)
    """
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
                if (rho > 1e-6) else
                np.array([[-1, 0],
                          [1, -1],
                          [-1, 0]]))

class CartesianDynamicsWrapper:
    def __init__(self, x_goal, polar_dynamics = PolarDynamics()):
        self.x_goal = x_goal
        self.polar_dynamics = polar_dynamics

    def f(self, x):
        return polar2cartesian(self.polar_dynamics.f(cartesian2polar(x, self.x_goal)))

    def g(self, x):
        return polar2cartesian(self.polar_dynamics.g(cartesian2polar(x, self.x_goal)))


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


class CLFPolar:
    def __init__(self,
                 Kp = np.array([9, 15, 5])/10.):
        self.Kp = np.asarray(Kp)

    def clf_terms(self, polar):
        return self._clf_terms(polar)

    def _clf_terms(self, polar):
        rho, alpha, beta = polar
        return np.array((0.5 * self.Kp[0] * rho ** 2,
                         self.Kp[1] * (1-np.cos(alpha)),
                         self.Kp[2] * (1-np.cos(alpha - beta))
        ))

    def grad_clf(self, polar):
        return self._grad_clf_terms(polar).sum(axis=-1)

    def _grad_clf_terms(self, polar):
        """
        >>> self = CLFPolar()
        >>> x0 = np.random.rand(3)
        >>> ajac = self._grad_clf_terms(x0).sum(axis=-1)
        >>> njac = numerical_jac(lambda x: self._clf_terms(x).sum(), x0, 1e-6)[0]
        >>> np.testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        rho, alpha, beta = polar
        return np.array([[self.Kp[0] * rho,  0, 0],
                         [0, self.Kp[1] * np.sin(alpha),
                          self.Kp[2] * np.sin(alpha - beta)] ,
                         [0,  0, - self.Kp[2] * np.sin(alpha - beta)]])

    def isconverged(self, x, x_goal):
        rho, alpha, beta = cartesian2polar(x, x_goal)
        return rho < 1e-3


def numerical_jac(func, x0, eps):
    """
    >>> def func(x): return np.array([np.cos(x[0]), np.sin(x[1])])
    >>> def jacfunc(x): return np.array([[-np.sin(x[0]), 0], [0, np.cos(x[1])]])
    >>> x0 = np.random.rand(2)
    >>> njac = numerical_jac(func, x0, 1e-6)
    >>> ajac = jacfunc(x0)
    >>> np.testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
    """
    f0 = func(x0)
    m = 1 if np.isscalar(f0) else f0.shape[-1]
    jac = np.empty((m, x0.shape[-1]))
    Dx = eps * np.eye(x0.shape[-1])
    XpDx = x0 + Dx
    for c in range(x0.shape[-1]):
        jac[:, c:c+1] = (func(XpDx[c, :]).reshape(-1, 1) - f0.reshape(-1, 1)) / eps

    return jac


class CLFCartesian:
    def __init__(self,
                 Kp = np.array([9, 15, 5])/10.):
        self.Kp = np.asarray(Kp)

    def clf_terms(self, state, state_goal):
        rho, alpha, beta = cartesian2polar(state, state_goal)
        x,y, theta = state
        x_goal, y_goal, theta_goal = state_goal
        return np.array((0.5 * self.Kp[0] * rho ** 2,
                         self.Kp[1] * cosdist(alpha, 0),
                         self.Kp[2] * cosdist(theta_goal, theta)
        ))

    def _grad_clf_terms(self, state, state_goal):
        """
        >>> self = CLFCartesian()
        >>> x0 = np.random.rand(3)
        >>> x0_goal = np.random.rand(3)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 0]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x, x0_goal)[0], x0, 1e-6)[0]
        >>> np.testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 1]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x, x0_goal)[1], x0, 1e-6)[0]
        >>> np.testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        >>> ajac = self._grad_clf_terms(x0, x0_goal)[:, 2]
        >>> njac = numerical_jac(lambda x: self.clf_terms(x, x0_goal)[2], x0, 1e-6)[0]
        >>> np.testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        x_diff, y_diff, theta_diff = state_goal - state
        rho, alpha, beta = cartesian2polar(state, state_goal)
        return np.array([[- self.Kp[0] * x_diff,
                          self.Kp[1] * np.sin(alpha) * y_diff / (rho**2),
                          0],
                         [- self.Kp[0] * y_diff,
                          - self.Kp[1] * np.sin(alpha) * x_diff / (rho**2),
                         0],
                         [0,
                          -self.Kp[1] * np.sin(alpha),
                          - self.Kp[2] * np.sin(theta_diff)]
                         ])
    def grad_clf(self, state, state_goal):
        """
        >>> self = CLFCartesian()
        >>> x0 = np.random.rand(3)
        >>> x0_goal = np.random.rand(3)
        >>> ajac = self.grad_clf(x0, x0_goal)
        >>> njac = numerical_jac(lambda x: self.clf_terms(x, x0_goal).sum(), x0, 1e-6)[0]
        >>> np.testing.assert_allclose(njac, ajac, rtol=1e-3, atol=1e-4)
        """
        return self._grad_clf_terms(state, state_goal).sum(axis=-1)

    def isconverged(self, x, x_goal):
        rho, alpha, beta = cartesian2polar(x, x_goal)
        return rho < 1e-3


class ControllerCLF:
    """
    Aicardi, M., Casalino, G., Bicchi, A., & Balestrino, A. (1995). Closed loop steering of unicycle like vehicles via Lyapunov techniques. IEEE Robotics & Automation Magazine, 2(1), 27-35.
    """
    def __init__(self, # simulation parameters
                 u_dim = 2,
                 coordinate_converter = cartesian2polar,
                 dynamics = PolarDynamics(),
                 clf = CLFPolar()):
        self.u_dim = 2
        self.coordinate_converter = coordinate_converter
        self.dynamics = dynamics
        self.clf = clf

    def _clf(self, polar):
        return self.clf.clf_terms(polar).sum()

    def _grad_clf(self, polar):
        return self.clf.grad_clf(polar)

    def _clc(self, x, x_goal, u, t):
        polar = self.coordinate_converter(x, x_goal)
        f, g = self.dynamics.f, self.dynamics.g
        gclf = self._grad_clf(polar)
        LOG.add_scalar("x_0", x[0], t)
        print("x :", x)
        print("clf terms :", self.clf.clf_terms(polar))
        print("clf:", self.clf.clf_terms(polar).sum())
        print("grad_x clf:", gclf)
        print("g(x): ", g(polar))
        print("grad_u clf:", gclf @ g(polar))
        return gclf @ (f(polar) + g(polar) @ u) + 10 * self._clf(polar)

    def _cost(self, x, u):
        import cvxpy as cp # pip install cvxpy
        return cp.sum_squares(u)

    def control(self, x, x_goal, t):
        import cvxpy as cp # pip install cvxpy
        x = np.asarray(x)
        uvar = cp.Variable(self.u_dim)
        uvar.value = np.zeros(self.u_dim)
        relax = cp.Variable(1)
        obj = cp.Minimize(self._cost(x, uvar) + 10*self._clc(x, x_goal, uvar, t))
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


    def isconverged(self, state, state_goal):
        return self.clf.isconverged(state, state_goal)


class ControllerPID:
    def __init__(self, # simulation parameters
                 Kp_rho = 9,
                 Kp_alpha = 15,
                 Kp_beta = -3):
        self.Kp_rho = Kp_rho
        self.Kp_alpha = Kp_alpha
        self.Kp_beta = Kp_beta

    def control(self, x, x_goal, t):
        rho, alpha, beta = cartesian2polar(x, x_goal)
        Kp_rho   = self.Kp_rho
        Kp_alpha = self.Kp_alpha
        Kp_beta  = self.Kp_beta
        v = Kp_rho * rho
        w = Kp_alpha * alpha + Kp_beta * beta
        if alpha > np.pi / 2 or alpha < -np.pi / 2:
            v = -v
        return [v, w]

    def isconverged(self, x, x_goal):
        rho, alpha, beta = cartesian2polar(x, x_goal)
        return rho < 1e-3


def move_to_pose(state_start, state_goal,
                 dt = 0.01,
                 show_animation = True,
                 controller=ControllerCLF(),
                 dynamics=CartesianDynamics()):
    """
    rho is the distance between the robot and the goal position
    alpha is the angle to the goal relative to the heading of the robot
    beta is the angle between the robot's position and the goal position plus the goal angle

    Kp_rho*rho and Kp_alpha*alpha drive the robot along a line towards the goal
    Kp_beta*beta rotates the line so that it is parallel to the goal angle
    """

    x_traj, y_traj = [], []

    state = state_start.copy()
    count = 0
    while not controller.isconverged(state, state_goal):
        x, y, theta = state
        x_traj.append(x)
        y_traj.append(y)

        # control
        ctrl = controller.control(state, state_goal, t=count)

        # simulation
        state = state + (dynamics.f(state) + dynamics.g(state) @ ctrl) * dt

        # visualization
        if show_animation:  # pragma: no cover
            plt.cla()
            x_start, y_start, theta_start = state_start
            plt.arrow(x_start, y_start, np.cos(theta_start),
                      np.sin(theta_start), color='r', width=0.1)
            x_goal, y_goal, theta_goal = state_goal
            plt.arrow(x_goal, y_goal, np.cos(theta_goal),
                      np.sin(theta_goal), color='g', width=0.1)
            plot_vehicle(x, y, theta, x_traj, y_traj, dt)
        count = count + 1


def plot_vehicle(x, y, theta, x_traj, y_traj, dt):  # pragma: no cover
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
            lambda event: [sys.exit(0) if event.key == 'escape' else None])

    plt.xlim(-2, 22)
    plt.ylim(-2, 22)

    plt.pause(dt)


def transformation_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])


class Configs:
    @property
    def clf_polar(self):
        return dict(simulator=partial(
            lambda x, x_g, **kw: move_to_pose(
                x , x_g,
                dynamics=CartesianDynamics(),
                **kw),
            controller=ControllerCLF(
                coordinate_converter = cartesian2polar,
                dynamics=PolarDynamics(),
                clf = CLFPolar()
            )))

    @property
    def clf_cartesian(self):
        return dict(simulator=partial(move_to_pose,
                                      dynamics=CartesianDynamics(),
                                      controller=ControllerCLF(
                                          dynamics=CartesianDynamics(),
                                          clf = CLFCartesian()
                                      )))

    @property
    def pid(self):
        return dict(simulator=partial(move_to_pose,
                                      dynamics=CartesianDynamics(),
                                      controller=ControllerPID()))


def main(simulator = move_to_pose):
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
        simulator(np.array([x_start, y_start, theta_start]),
                  np.array([x_goal, y_goal, theta_goal]))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    main(**getattr(Configs(), 'clf_polar')) # 'pid', 'clf_polar' or 'clf_cartesian'
