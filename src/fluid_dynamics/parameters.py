from typing import List

import numpy as np
from pydantic import BaseModel, Field

from src.geometry import DomainGeometry
from src.constants import G
from src.utils import FileMixin


class FluidParameters(BaseModel, FileMixin):
    u_pt: float = Field(..., gt=0.0, description="Phase transition temperature [K].")
    u_ref: float = Field(..., gte=0.0, description="Reference temperature [K].")
    delta_u: float = Field(
        ..., gt=0.0, description="Characteristic temperature difference [K]."
    )
    v: float = Field(..., gt=0.0, description="Characteristic flow velocity [m/s].")
    epsilon: float = Field(
        ...,
        gt=0.0,
        description="Parameter of the indicator function used in the fictitious domain method.",
    )
    kinematic_viscosity_coeffs: List[float] = Field(
        ...,
        description="Polynomial coefficients for kinematic viscosity at reference temperature. "
        "The first element is the coefficient for u_ref^0, the second for u_ref^1, etc.",
    )
    volumetric_thermal_exp_coeffs: List[float] = Field(
        ...,
        description="Polynomial coefficients for volumetric thermal expansion coefficient at reference temperature. "
        "The first element is the coefficient for u_ref^0, the second for u_ref^1, etc.",
    )

    @property
    def kinematic_viscosity_at_u_ref(self) -> float:
        """
        Calculate the kinematic viscosity coefficient at the reference temperature using polynomial evaluation.
        """
        return np.polyval(self.kinematic_viscosity_coeffs[::-1], self.u_ref)

    @property
    def thermal_exp_coefficient_at_u_ref(self) -> float:
        """
        Calculate the volumetric thermal expansion coefficient at the reference temperature using polynomial evaluation.
        """
        return np.polyval(self.volumetric_thermal_exp_coeffs[::-1], self.u_ref)

    @property
    def u_pt_ref(self) -> float:
        """
        Calculate the deviation of phase transition temperature from the reference temperature.
        """
        return self.u_pt - self.u_ref

    @property
    def reynolds_number(self) -> float:
        """
        Calculate the Reynolds number at the reference temperature.
        Formula: Re = characteristic_length * flow_velocity / kinematic_viscosity
        """
        return (
            self.v
            * self.domain_geometry.length_scale
            / self.kinematic_viscosity_at_u_ref
        )

    @property
    def grashof_number(self) -> float:
        """
        Calculate the Grashof number at the reference temperature.
        Formula: Gr = g * thermal_expansion_coefficient * delta_u * l^3 / kinematic_viscosity^2
        """
        l = self.domain_geometry.length_scale
        return (
            G
            * self.thermal_exp_coefficient_at_u_ref
            * self.delta_u
            * l
            * l
            * l
            / (self.kinematic_viscosity_at_u_ref * self.kinematic_viscosity_at_u_ref)
        )

    def __str__(self):
        s = (
            f"Fluid Parameters:\n"
            f"  Phase Transition Temperature: {self.u_pt} K\n"
            f"  Reference Temperature: {self.u_ref} K\n"
            f"  Parameter of the Indicator Function (Epsilon): {self.epsilon}\n"
            f"  Kinematic Viscosity at the Reference Temperature (Water): "
            f"{self.kinematic_viscosity_at_u_ref:.2E} m^2/s\n"
            f"  Volumetric Thermal Expansion Coefficient at the Reference Temperature (Water): "
            f"{self.thermal_exp_coefficient_at_u_ref:.2E} 1/K\n"
            f"  Characteristic Flow Velocity {self.v} m/s\n"
            f"  Characteristic Temperature Difference {self.delta_u:.2E} K\n"
            f"  Reynolds number {self.reynolds_number:.2E}\n"
            f"  Grashof number {self.grashof_number:.2E}\n"
        )
        return s
