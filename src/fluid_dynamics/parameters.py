from pydantic import BaseModel, Field

from src.geometry import DomainGeometry
from src.constants import G


class FluidParameters(BaseModel):
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
    domain_geometry: DomainGeometry

    @property
    def kinematic_viscosity_at_u_ref(self) -> float:
        """
        Calculate the kinematic viscosity coefficient at the reference temperature (water).
        """
        return (
            -2.54761652e-12 * self.u_ref * self.u_ref * self.u_ref
            + 2.65889022e-09 * self.u_ref * self.u_ref
            - 9.28722151e-07 * self.u_ref
            + 1.08963453e-04
        )

    @property
    def thermal_exp_coefficient_at_u_ref(self) -> float:
        """
        Calculate the volumetric thermal expansion coefficient at the reference temperature (water).
        """
        return (
            -9.84848485e-08 * self.u_ref * self.u_ref
            + 6.86739177e-05 * self.u_ref
            - 1.14630054e-02
        )

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
            f"  Characteristic Temperature Difference {self.delta_u} K\n"
            f"  Reynolds number {self.reynolds_number:.2E}\n"
            f"  Grashof number {self.grashof_number:.2E}\n"
        )
        return s
