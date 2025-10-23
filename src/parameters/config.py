from functools import cached_property
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.core.constants import G
from src.core.geometry import DomainGeometry
from src.parameters.material_properties import MaterialProperties
from src.parameters.mixins import FileIOMixin


class ExperimentConfig(BaseModel, FileIOMixin):
    # — Core geometry —
    geometry: DomainGeometry = Field(..., description="Domain geometry")

    # — Experiment characteristics —
    u_ref: float = Field(..., gt=0.0, description="Reference temperature [K].")
    delta_u: float = Field(
        ..., gt=0.0, description="Characteristic temperature difference [K]."
    )
    v: float = Field(..., gt=0.0, description="Characteristic flow velocity [m/s].")
    l: float = Field(..., gt=0.0, description="Characteristic length [m].")

    # — Smoothing parameters —
    delta: Optional[float] = Field(
        None, gt=0.0, description="Half of the mushy zone temperature range [K]."
    )
    epsilon: float = Field(
        ...,
        gt=0.0,
        description="Parameter of the indicator function used in the fictitious domain method.",
    )

    # — Material properties —
    material_props: MaterialProperties = Field(..., description="Material properties")

    # — output control —
    output_dir: Optional[DirectoryPath] = Field(
        None, description="Where to dump results"
    )
    save_interval: Optional[PositiveInt] = Field(
        None, description="Steps between saves"
    )

    @cached_property
    def scaled_grid_steps(self) -> Tuple[float, float, float]:
        """
        Calculate nondimensionalized space and time grid steps.
        :return: dx_scaled, dy_scaled, dt_scaled
        """
        dx, dy, dt = self.geometry.dx, self.geometry.dy, self.geometry.dt
        return dx / self.l, dy / self.l, dt * self.v / self.l

    @property
    def u_pt_ref(self) -> float:
        """
        Calculate the deviation of phase transition temperature from the reference temperature.
        """
        return self.material_props.u_pt - self.u_ref

    @property
    def u_pt_nd(self) -> float:
        """
        Calculate the nondimensionalized phase transition temperature.
        """
        return (self.material_props.u_pt - self.u_ref) / self.delta_u

    @property
    def delta_nd(self) -> Optional[float]:
        """
        1/2 of nondimensional mushy zone temperature range.
        """
        if self.delta is not None:
            return self.delta / self.delta_u
        return None

    @cached_property
    def volumetric_heat_capacity_ref(self) -> float:
        """
        Calculate the volumetric heat capacity at the reference temperature.
        """
        if self.u_ref < self.material_props.u_pt:
            return self.material_props.volumetric_heat_capacity_solid
        elif self.u_ref > self.material_props.u_pt:
            return self.material_props.volumetric_heat_capacity_liquid

        return self.material_props.volumetric_heat_capacity_liquid
        # return 0.5 * (
        #     self.material_props.volumetric_heat_capacity_solid
        #     + self.material_props.volumetric_heat_capacity_liquid
        # )

    @cached_property
    def thermal_conductivity_ref(self) -> float:
        """
        Calculate the thermal conductivity at the reference temperature.
        """
        if self.u_ref < self.material_props.u_pt:
            return self.material_props.thermal_conductivity_solid
        elif self.u_ref > self.material_props.u_pt:
            return self.material_props.thermal_conductivity_liquid
        return self.material_props.thermal_conductivity_liquid
        # return 0.5 * (
        #     self.material_props.thermal_conductivity_solid
        #     + self.material_props.thermal_conductivity_liquid
        # )

    @cached_property
    def thermal_diffusivity_ref(self) -> float:
        """
        Calculate the thermal diffusivity at the reference temperature.
        """
        return self.thermal_conductivity_ref / self.volumetric_heat_capacity_ref

    @cached_property
    def kinematic_viscosity_ref(self) -> float:
        """
        Calculate the kinematic viscosity coefficient at the reference temperature using polynomial evaluation.
        """
        return np.polyval(
            self.material_props.kinematic_viscosity_coeffs[::-1], self.u_ref
        )

    @cached_property
    def thermal_exp_coefficient_ref(self) -> float:
        """
        Calculate the volumetric thermal expansion coefficient at the reference temperature using polynomial evaluation.
        """
        return np.polyval(
            self.material_props.volumetric_thermal_exp_coeffs[::-1], self.u_ref
        )

    @cached_property
    def peclet_number(self) -> float:
        """
        Calculate the Péclet number at the reference temperature.
        Formula: Pe = characteristic_length * flow_velocity *  volumetric_heat_capacity / thermal_conductivity
        """
        return (
            self.v
            * self.l
            * self.volumetric_heat_capacity_ref
            / self.thermal_conductivity_ref
        )

    @cached_property
    def stefan_number(self) -> float:
        """
        Calculate the Stefan number for liquid phase.
        Formula: Ste = specific_heat_liquid * temperature_difference / specific_latent_heat
        """
        return (
            self.volumetric_heat_capacity_ref
            * self.delta_u
            / self.material_props.volumetric_latent_heat
        )

    @cached_property
    def reynolds_number(self) -> float:
        """
        Calculate the Reynolds number at the reference temperature.
        Formula: Re = characteristic_length * flow_velocity / kinematic_viscosity
        """
        return self.v * self.l / self.kinematic_viscosity_ref

    @cached_property
    def grashof_number(self) -> float:
        """
        Calculate the Grashof number at the reference temperature.
        Formula: Gr = g * thermal_expansion_coefficient * delta_u * l^3 / kinematic_viscosity^2
        """
        return (
            G
            * abs(self.thermal_exp_coefficient_ref)
            * self.delta_u
            * self.l
            * self.l
            * self.l
            / (self.kinematic_viscosity_ref * self.kinematic_viscosity_ref)
        )

    @cached_property
    def prandtl_number(self) -> float:
        """
        Calculate the Prandtl number at the reference temperature.
        Formula: Pr = kinematic_viscosity / thermal_diffusivity
        """
        return self.kinematic_viscosity_ref / self.thermal_diffusivity_ref

    @cached_property
    def rayleigh_number(self) -> float:
        """
        Calculate the Rayleigh number at the reference temperature.
        Formula: Ra = Gr * Pr
        """
        return self.grashof_number * self.prandtl_number

    model_config = {"populate_by_name": True, "extra": "ignore"}

    def __str__(self):
        exp_char = (
            f"Characteristic values:\n"
            f"  Reference Temperature: {self.u_ref} K\n"
            f"  Characteristic Temperature Difference {self.delta_u:.2E} K\n"
            f"  Characteristic Flow Velocity {self.v} m/s\n"
            f"  Characteristic Length {self.l} m\n"
        )
        dim_nums = (
            f"Dimensionless numbers:\n"
            f"  Re = {self.reynolds_number:.3E}\n"
            f"  Gr = {self.grashof_number:.3E}\n"
            f"  Ra = {self.rayleigh_number:.3E}\n"
            f"  Ste = {self.stefan_number:.3E}\n"
            f"  Pe = {self.peclet_number:.3E}\n"
            f"  Pr = {self.prandtl_number:.3E}\n"
        )
        return (
            str(self.geometry)
            + "\n"
            + str(self.material_props)
            + "\n"
            + exp_char
            + "\n"
            + dim_nums
        )
