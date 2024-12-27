import numpy as np
from numpy.typing import NDArray

from src.boundary_conditions import BoundaryCondition
from src.fluid_dynamics.parameters import FluidParameters
from src.fluid_dynamics.solvers.registry import (
    NavierStokesSchemeName,
    NavierStokesSchemeRegistry,
)
from src.fluid_dynamics.solvers.schemes import *  # noqa, automatically register all of the schemes
from src.geometry import DomainGeometry


class NavierStokesSolver:
    def __init__(
        self,
        scheme: NavierStokesSchemeName,
        geometry: DomainGeometry,
        parameters: FluidParameters,
        top_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
        bottom_bc: BoundaryCondition,
        left_bc: BoundaryCondition,
        sf_max_iters: int = 50,
        sf_stopping_criteria: float = 1e-6,
        implicit_lin_max_iters: int = 5,
        implicit_lin_stopping_criteria: float = 1e-6,
        implicit_lin_urf: float = 0.5,
    ):
        self.scheme = scheme
        scheme_class = NavierStokesSchemeRegistry.get_scheme_class(self.scheme)
        self.solver = scheme_class(
            geometry=geometry,
            parameters=parameters,
            top_bc=top_bc,
            right_bc=right_bc,
            bottom_bc=bottom_bc,
            left_bc=left_bc,
            sf_max_iters=sf_max_iters,
            sf_stopping_criteria=sf_stopping_criteria,
            implicit_lin_max_iters=implicit_lin_max_iters,
            implicit_lin_stopping_criteria=implicit_lin_stopping_criteria,
            implicit_lin_urf=implicit_lin_urf,
        )

    def solve(
        self,
        w: NDArray[np.float64],
        sf: NDArray[np.float64],
        u: NDArray[np.float64],
        time: float = 0.0,
    ):
        return self.solver.solve(w, sf, u, time)
