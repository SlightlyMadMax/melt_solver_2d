from pydantic import BaseModel, Field, validator

from src.constants import ABS_ZERO


class FluidParameters(BaseModel):
    u_pt: float = Field(..., gt=0.0, description="Phase transition heat_transfer [K].")
    u_ref: float = Field(..., gte=0.0, description="Reference heat_transfer [K].")
    epsilon: float = Field(
        ...,
        gt=0.0,
        description="Parameter of the indicator function used in the fictitious domain method.",
    )

    @property
    def kinematic_viscosity_at_u_ref(self) -> float:
        """
        Calculate the kinematic viscosity coefficient at the reference heat_transfer.
        """
        return (
            -2.74319393e-12
            * (self.u_ref + ABS_ZERO)
            * (self.u_ref + ABS_ZERO)
            * (self.u_ref + ABS_ZERO)
            + 6.03737170e-10 * (self.u_ref + ABS_ZERO) * (self.u_ref + ABS_ZERO)
            - 4.77142009e-08 * (self.u_ref + ABS_ZERO)
            + 1.75308187e-06
        )

    @property
    def u_pt_ref(self) -> float:
        """
        Calculate the deviation of phase transition heat_transfer from the reference heat_transfer.
        """
        return self.u_pt - self.u_ref

    def __str__(self):
        s = (
            f"Fluid Parameters:\n"
            f"  Phase Transition Temperature: {self.u_pt} K\n"
            f"  Reference Temperature: {self.u_ref} K\n"
            f"  Parameter of the Indicator Function (Epsilon): {self.epsilon}\n"
            f"  Kinematic Viscosity at the Reference Temperature: {self.kinematic_viscosity_at_u_ref} m^2/s\n"
        )
        return s
