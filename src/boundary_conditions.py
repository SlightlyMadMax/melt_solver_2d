from enum import Enum
import numpy as np
from typing import Callable


class BoundaryConditionType(Enum):
    DIRICHLET = 1
    NEUMANN = 2
    ROBIN = 3


class BoundaryCondition:
    def __init__(
        self,
        boundary_type: BoundaryConditionType,
        n: int,
        value_func: Callable[[float, int], np.ndarray] = None,
        flux_func: Callable[[float, int], np.ndarray] = None,
        phi: Callable[[float, int], np.ndarray] = None,
        psi: Callable[[float, int], np.ndarray] = None,
    ):
        """
        Initialize a boundary condition.

        Neumann condition is expected to be in the form of u' = f(t), where f is a flux.
        Robin condition is expected to be in the form of u' = phi(t) * u + psi(t).
        :param boundary_type: Type of boundary condition.
        :param n: Number of nodes on the boundary.
        :param value_func: Callable function for Dirichlet condition (returns boundary values as a ndarray at a given time).
        :param flux_func: Callable function for Neumann condition (returns flux values as a ndarray at a given time).
        :param phi: Callable function for the psi parameter (Robin condition only).
        :param psi: Callable function for the psi parameter (Robin condition only).
        """
        valid_types = set(t for t in BoundaryConditionType)
        assert (
            boundary_type in valid_types
        ), f"Invalid boundary type. Must be one of {valid_types}."
        self.boundary_type = boundary_type
        self.n = n

        self._validate_callable("value_func", value_func, n)
        self._validate_callable("flux_func", flux_func, n)
        self._validate_callable("phi", phi, n)
        self._validate_callable("psi", psi, n)

        self.value_func = value_func
        self.flux_func = flux_func
        self.phi = phi
        self.psi = psi

        self._validate_boundary_type()

    @staticmethod
    def _validate_callable(name: str, func: Callable, n: int) -> None:
        if func is not None:
            if not callable(func):
                raise TypeError(f"{name} must be callable.")
            test_output = func(0.0, n)
            if not isinstance(test_output, np.ndarray):
                raise TypeError(f"{name} must return a numpy ndarray.")
            if test_output.shape[0] != n:
                raise ValueError(f"{name} must return a numpy ndarray of length {n}.")

    def _validate_boundary_type(self):
        if (
            self.boundary_type == BoundaryConditionType.DIRICHLET
            and self.value_func is None
        ):
            raise ValueError(
                "Dirichlet boundary condition requires 'value_func' to provide boundary values."
            )
        if (
            self.boundary_type == BoundaryConditionType.NEUMANN
            and self.flux_func is None
        ):
            raise ValueError(
                "Neumann boundary condition requires 'flux_func' to provide flux values."
            )
        if self.boundary_type == BoundaryConditionType.ROBIN and (
            self.phi is None or self.psi is None
        ):
            raise ValueError(
                "Robin boundary condition requires 'phi' and 'psi' to define the boundary behavior."
            )

    def get_value(self, t: float) -> np.ndarray:
        """
        Get the boundary value for Dirichlet condition at a specific time as a ndarray.

        :param t: Time at which the boundary value is required.
        :return: ndarray of boundary condition values.
        """
        assert (
            self.boundary_type == BoundaryConditionType.DIRICHLET
        ), "get_value is only valid for Dirichlet condition"
        return self.value_func(t, self.n)

    def get_flux(self, t: float) -> np.ndarray:
        """
        Get the flux values for Neumann condition at a specific time as a ndarray.

        :param t: Time at which the flux values are required.
        :return: ndarray of flux values for Neumann condition.
        """
        assert (
            self.boundary_type == BoundaryConditionType.NEUMANN
        ), "get_flux is only valid for Neumann condition"
        return self.flux_func(t, self.n)

    def get_phi(self, t: float) -> np.ndarray:
        """
        Get the phi parameter value at a specific time.

        :param t: Time at which the flux values are required.
        :return: ndarray of phi values for Robin condition.
        """
        return self.phi(t, self.n)

    def get_psi(self, t: float) -> np.ndarray:
        """
        Get the psi parameter value at a specific time.

        :param t: Time at which the flux values are required.
        :return: ndarray of phi values for Robin condition.
        """
        return self.psi(t, self.n)
