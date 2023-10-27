from dataclasses import dataclass
from typing import Callable

import nlopt
import numpy as np

from frogger.robots.robot_core import RobotModel
from frogger.sampling import ICSampler


@dataclass
class FroggerConfig:
    """Configuration for the Frogger solver."""
    model: RobotModel
    sampler: ICSampler
    tol_surf: float = 1e-3
    tol_joint: float = 1e-2
    tol_col: float = 1e-3
    tol_fclosure: float = 1e-5
    tol_couple: float = 1e-6
    xtol_rel: float = 1e-6
    xtol_abs: float = 1e-6
    maxeval: int = 1000
    maxtime: float | None = 60.0

class Frogger:
    """The FRoGGeR solver."""

    def __init__(self, cfg: FroggerConfig) -> None:
        """Initializes the solver."""
        # model parameters
        model = cfg.model
        n = model.n
        nc = model.nc  # number of contacts
        mu = model.mu
        ns = model.ns
        n_bounds = model.n_bounds
        n_couple = model.n_couple

        # solver settings
        tol_surf = cfg.tol_surf
        tol_joint = cfg.tol_joint
        tol_col = cfg.tol_col
        tol_fclosure = cfg.tol_fclosure
        tol_couple = cfg.tol_couple
        xtol_rel = cfg.xtol_rel
        xtol_abs = cfg.xtol_abs
        maxeval = cfg.maxeval
        maxtime = cfg.maxtime

        # constraint setup
        n_joint = model.n_bounds
        n_col = len(model.query_object.inspector().GetCollisionCandidates())
        n_surf = nc

        n_ineq = n_joint + n_col + 1
        n_eq = n_surf + n_couple

        tol_eq = tol_surf * np.ones(n_eq)  # surface constraint tolerances
        tol_eq[n_surf:(n_surf + n_couple)] = tol_couple
        tol_ineq = tol_col * np.ones(n_ineq)  # collision constraint tolerances
        tol_ineq[:n_joint] = tol_joint  # joint limit constraint tolerances
        tol_ineq[n_joint + n_col] = tol_fclosure  # fclosure constraint tolerance

        # setting up the solver
        f, g, h = self._make_fgh(model)
        opt = nlopt.opt(nlopt.LD_SLSQP, model.n)
        opt.set_xtol_rel(xtol_rel)
        opt.set_xtol_abs(xtol_abs)
        opt.set_maxeval(maxeval)
        opt.set_min_objective(f)
        opt.add_inequality_mconstraint(g, tol_ineq)
        opt.add_equality_mconstraint(h, tol_eq)
        if maxtime is not None:
            opt.set_maxtime(maxtime)

        # setting attributes
        self.opt = opt
        self.model = model
        self.sampler = cfg.sampler
        self.f, self.g, self.h = f, g, h
        self.n_ineq, self.n_eq = n_ineq, n_eq
        self.n_surf, self.n_joint, self.n_couple = n_surf, n_joint, n_couple
        self.tol_surf = tol_surf
        self.tol_joint = tol_joint
        self.tol_col = tol_col
        self.tol_fclosure = tol_fclosure
        self.tol_couple = tol_couple

    def _make_fgh(self, model) -> tuple[Callable, Callable, Callable]:
        """Returns f, g, and h suitable for use in NLOPT.

        We do not use the Drake NLOPT wrapper because of the overhead required to
        specify gradients that are not autodifferentiated by Drake, which we must cast
        into AutoDiffXd objects. We measured the conversion time and concluded it was
        significant enough to just use the Python implementation of NLOPT directly.

        Returns
        -------
        f : Callable[[np.ndarray, np.ndarray], float]
            Cost function.
        g : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            Vector inequality constraints.
        h : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            Vector equality constraints.
        """
        def f(q, grad):
            if grad.size > 0:
                grad[:] = model.compute_Df(q)
            return model.compute_f(q)

        def g(result, q, grad):
            if grad.size > 0:
                grad[:] = model.compute_Dg(q)
            result[:] = model.compute_g(q)

        def h(result, q, grad):
            if grad.size > 0:
                grad[:] = model.compute_Dh(q)
            result[:] = model.compute_h(q)

        return f, g, h

    def generate_grasp(self) -> np.ndarray:
        """Generates a grasp."""
        success = False
        while not success:
            # sample initial guess
            q0, _ = self.sampler.sample_configuration()

            # refine the grasp
            try:
                q_star = self.opt.optimize(q0)
            except (RuntimeError, ValueError, nlopt.RoundoffLimited):
                # [NOTE] RuntimeError catches two extremely rare errors:
                #        "RuntimeError: bug: more than iter SQP iterations"
                #          -this is an NLOPT error
                #        "RuntimeError: Error with configuration"
                #          -see: github.com/RobotLocomotion/drake/issues/18704
                # [NOTE] ValueError catches a rare error involving nans appearing in
                #        MeshObject gradient computation
                q_star = np.nan * np.ones(self.model.n)

            # check whether the solution satisfies the constraints
            if np.any(np.isnan(q_star)):
                continue  # nan means an error - resample
            else:
                # computing f, g, h at q_star
                f_val = self.f(q_star, np.zeros(0))
                g_val = np.zeros(self.n_ineq)
                self.g(g_val, q_star, np.zeros(0))
                h_val = np.zeros(self.n_eq)
                self.h(h_val, q_star, np.zeros(0))

                # checking whether the computed solution is feasible
                surf_vio = np.max(np.abs(h_val[:self.n_surf]))
                couple_vio = np.max(
                    np.abs(h_val[self.n_surf:(self.n_surf + self.n_couple)])
                )
                joint_vio = max(np.max(g_val[:self.model.n_bounds]), 0.0)
                col_vio = max(np.max(g_val[self.model.n_bounds:-1]), 0.0)
                fclosure_vio = max(g_val[-1], 0.0)

                # setting the feasibility flag
                success = (
                    surf_vio <= self.tol_surf
                    and couple_vio <= self.tol_couple
                    and joint_vio <= self.tol_joint
                    and col_vio <= self.tol_col
                    and fclosure_vio <= self.tol_fclosure  # min-weight bound
                )
        return q_star
