"""
Domain energy budget of the Peaceman-Rachford heat step.

The books are kept the way the scheme itself sees the energy, so that what is left
over is genuine non-conservation of the discretisation rather than a mismatch between
the diagnostic and the operator.

The heat equation is advanced in the form

    u_t + v . grad u = div(k grad u) / (Pe c_eff),     c_eff, k taken at u^n.

Adding both half-steps and multiplying by c_eff^n gives, at every interior node,

    c^n (u^{n+1} - u^n) = tau [ div_h(k grad_h u) - c^n (advection) ],

with the x-diffusion evaluated at u^{n+1/2} (implicit in the first half-step, explicit
in the second) and the y-diffusion averaged over u^n and u^{n+1}. Over the interior
nodes the diffusion telescopes into the heat crossing the walls, Q, so the change of
the model enthalpy H = H_sens + H_lat splits as

    dE - tau Q = [dE - sum c^n du] + [sum c^n du - tau Q]
                 \\______________/   \\__________________/
                    capacity             advection

capacity   the enthalpy change a coefficient frozen at u^n misses over a finite du;
           it is concentrated where the smoothed latent heat turns over, and
           vanishes as tau -> 0
advection  energy created or destroyed by the non-conservative advective operator
           (first-order upwind with a limited deferred correction); identically
           zero without flow, which is how the diagnostic itself is verified

H is the enthalpy the model actually carries: the integral of the smoothed c_eff,
available in closed form for the erf step and Gaussian delta.
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import erf

from src.heat_transfer.coefficient_smoothing.coefficients import DeltaScheme, StepScheme
from src.heat_transfer.solvers.heat_transfer_solvers.peaceman_rachford import (
    PeacemanRachfordSolver,
)
from src.parameters.config import ExperimentConfig

_FIELDS = (
    "t",
    "E_sens",
    "E_lat",
    "Q_hot",
    "Q_cold",
    "Q_adiabatic",
    "defect_capacity",
    "defect_advection",
)


class EnergyLedger:
    """Per-step energy budget of the interior nodes, fed from the runner's step callback."""

    def __init__(self, cfg: ExperimentConfig, heat_solver):
        solver = getattr(heat_solver, "solver", heat_solver)
        if not isinstance(solver, PeacemanRachfordSolver):
            raise TypeError(
                "the budget reproduces the Peaceman-Rachford splitting term by term; "
                f"got {type(solver).__name__}"
            )
        if solver.step_scheme != StepScheme.ERF or solver.delta_scheme != DeltaScheme.GAUSS:
            raise NotImplementedError("closed-form enthalpy exists for erf/gauss only")
        if cfg.delta_nd is None:
            raise ValueError(
                "an adaptive smoothing width changes H(u) from step to step, so the "
                "enthalpy change between steps is not defined; pass --eps-t"
            )

        self.solver = solver
        props = cfg.material_props
        c_ref = cfg.volumetric_heat_capacity_ref
        self.c_s = props.volumetric_heat_capacity_solid / c_ref
        self.c_l = props.volumetric_heat_capacity_liquid / c_ref
        self.latent = 1.0 / cfg.stefan_number
        self.u_0 = cfg.u_pt_nd
        self.sigma = cfg.delta_nd
        self.dx, self.dy, self.tau = cfg.scaled_grid_steps
        self.inv_pe = 1.0 / cfg.peclet_number
        self.area = self.dx * self.dy

        # Dimensional scales, per unit depth
        self.energy_scale = c_ref * cfg.delta_u * cfg.l**2  # J/m
        self.power_scale = self.energy_scale * cfg.v / cfg.l  # W/m

        n_y, n_x = cfg.geometry.n_y, cfg.geometry.n_x
        self._half = np.empty((n_y, n_x))
        self._u_prev = np.empty((n_y, n_x))
        self._e_prev = None
        self._e_0 = None
        self._rows = []

        # The x-wall heat is carried by the intermediate level, which the solver never
        # exposes; take a copy of it on its way to the second sweep
        first_sweep = solver._after_first_sweep

        def capture(result, **kwargs):
            self._half[:, :] = result
            first_sweep(result=result, **kwargs)

        solver._after_first_sweep = capture

    # ------------------------------------------------------------------ model energy

    def enthalpy(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sensible and latent enthalpy per unit volume, measured from the melting point.

        With S the erf step and g the Gaussian of width sigma, c_eff = c_s + (c_l - c_s) S
        + g / Ste, and int S dz = z S + sigma^2 g.
        """
        z = u - self.u_0
        s = self.sigma
        step = 0.5 * (1.0 + erf(z / (np.sqrt(2.0) * s)))
        gauss = np.exp(-0.5 * (z / s) ** 2) / (np.sqrt(2.0 * np.pi) * s)
        sensible = self.c_s * z + (self.c_l - self.c_s) * (z * step + s * s * gauss)
        return sensible, self.latent * step

    def _totals(self, u: np.ndarray) -> tuple[float, float]:
        # Only interior nodes store energy: wall nodes are Dirichlet values and the
        # adiabatic rows are first-order copies of their neighbours
        sensible, latent = self.enthalpy(u[1:-1, 1:-1])
        return float(sensible.sum()) * self.area, float(latent.sum()) * self.area

    # ------------------------------------------------------------------ wall heat

    def _wall_x(self, u: np.ndarray) -> tuple[float, float]:
        """Heat entering through the left (hot) and right (cold) walls, per unit time."""
        k_x = self.solver._k_x  # k_x[:, i] is the face between nodes i-1 and i
        g = self.inv_pe * self.dy / self.dx
        rows = slice(1, -1)
        hot = g * float(np.sum(k_x[rows, 1] * (u[rows, 0] - u[rows, 1])))
        cold = g * float(np.sum(k_x[rows, -2] * (u[rows, -1] - u[rows, -2])))
        return hot, cold

    def _wall_y(self, u: np.ndarray) -> float:
        """Heat entering through the bottom and top walls; zero once u obeys the BC."""
        k_y = self.solver._k_y
        g = self.inv_pe * self.dx / self.dy
        cols = slice(1, -1)
        bottom = float(np.sum(k_y[1, cols] * (u[0, cols] - u[1, cols])))
        top = float(np.sum(k_y[-2, cols] * (u[-1, cols] - u[-2, cols])))
        return g * (bottom + top)

    # ------------------------------------------------------------------ bookkeeping

    def prime(self, u: np.ndarray) -> None:
        """Record the starting state; call once before the first step."""
        self._u_prev[:, :] = u
        self._e_prev = self._totals(u)
        self._e_0 = self._e_prev

    def record(self, u: np.ndarray, t: float) -> None:
        """Account for the step that has just produced ``u``.

        Must run before the next heat solve: c_eff, k_x and k_y still hold their
        values at u^n until then.
        """
        if self._e_prev is None:
            raise RuntimeError("EnergyLedger.prime() was not called")

        u_n = self._u_prev
        e_sens, e_lat = self._totals(u)
        d_energy = (e_sens - self._e_prev[0]) + (e_lat - self._e_prev[1])

        inner = (slice(1, -1), slice(1, -1))
        c_eff = self.solver._c_eff
        stored = float(np.sum(c_eff[inner] * (u[inner] - u_n[inner]))) * self.area

        hot, cold = self._wall_x(self._half)
        adiabatic = 0.5 * (self._wall_y(u_n) + self._wall_y(u))
        wall = hot + cold + adiabatic

        self._rows.append(
            (t, e_sens, e_lat, hot, cold, adiabatic, d_energy - stored, stored - self.tau * wall)
        )
        self._u_prev[:, :] = u
        self._e_prev = (e_sens, e_lat)

    # ------------------------------------------------------------------ output

    def as_arrays(self) -> dict:
        """Dimensional series: energies in J/m, heat rates in W/m, defects in J/m per step."""
        raw = np.asarray(self._rows, dtype=float).reshape(-1, len(_FIELDS))
        col = dict(zip(_FIELDS, raw.T))
        es, ps = self.energy_scale, self.power_scale
        return {
            "t": col["t"],
            "dt": np.full_like(col["t"], self.tau * es / ps),
            "E_sens": col["E_sens"] * es,
            "E_lat": col["E_lat"] * es,
            "E0_sens": np.float64(self._e_0[0] * es),
            "E0_lat": np.float64(self._e_0[1] * es),
            "Q_hot": col["Q_hot"] * ps,
            "Q_cold": col["Q_cold"] * ps,
            "Q_adiabatic": col["Q_adiabatic"] * ps,
            "defect_capacity": col["defect_capacity"] * es,
            "defect_advection": col["defect_advection"] * es,
        }

    def summary(self) -> dict:
        a = self.as_arrays()
        if a["t"].size == 0:
            return {}
        dt = a["dt"]
        cap = float(a["defect_capacity"].sum())
        adv = float(a["defect_advection"].sum())
        change = float(a["E_sens"][-1] + a["E_lat"][-1] - a["E0_sens"] - a["E0_lat"])
        # Gross heat through the domain: the net wall heat tends to the storage rate
        # and becomes a poor yardstick once the ice front slows down
        throughput = float(np.sum(0.5 * (np.abs(a["Q_hot"]) + np.abs(a["Q_cold"])) * dt))
        return {
            "energy_change_J_per_m": change,
            "wall_heat_J_per_m": float(
                np.sum((a["Q_hot"] + a["Q_cold"] + a["Q_adiabatic"]) * dt)
            ),
            "throughput_J_per_m": throughput,
            "defect_capacity_J_per_m": cap,
            "defect_advection_J_per_m": adv,
            "defect_total_rel_change": (cap + adv) / abs(change) if change else float("nan"),
            "defect_total_rel_throughput": (
                (cap + adv) / throughput if throughput else float("nan")
            ),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(path, **self.as_arrays())
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        return path
