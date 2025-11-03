from typing import List

from pydantic import BaseModel, Field


class MaterialProperties(BaseModel):
    u_pt: float = Field(..., gt=0.0, description="Phase transition temperature [K].")
    specific_heat_liquid: float = Field(
        ...,
        gt=0.0,
        description="Specific heat capacity of the liquid phase [J/(kg⋅K)].",
    )
    specific_heat_solid: float = Field(
        ..., gt=0.0, description="Specific heat capacity of the solid phase [J/(kg⋅K)]."
    )
    specific_latent_heat: float = Field(
        ...,
        gt=0.0,
        description="Specific latent heat of fusion [J/kg].",
    )
    density_liquid: float = Field(
        ..., gt=0, description="Density of the liquid phase [kg/m^3]."
    )
    density_solid: float = Field(
        ..., gt=0, description="Density of the solid phase [kg/m^3]."
    )
    thermal_conductivity_liquid: float = Field(
        ..., gt=0, description="Thermal conductivity of the liquid phase [W/(m⋅K)]."
    )
    thermal_conductivity_solid: float = Field(
        ..., gt=0, description="Thermal conductivity of the solid phase [W/(m⋅K)]."
    )
    kinematic_viscosity_coeffs: List[float] = Field(
        ...,
        min_length=1,
        description="Polynomial coefficients for kinematic viscosity at reference temperature. "
        "The first element is the coefficient for u_ref^0, the second for u_ref^1, etc.",
    )
    volumetric_thermal_exp_coeffs: List[float] = Field(
        ...,
        min_length=1,
        description="Polynomial coefficients for volumetric thermal expansion coefficient at reference temperature. "
        "The first element is the coefficient for u_ref^0, the second for u_ref^1, etc.",
    )

    @property
    def volumetric_heat_capacity_liquid(self) -> float:
        """
        Calculate the volumetric heat capacity for the liquid phase.
        Formula: volumetric_heat_capacity = density * specific_heat
        """
        return self.density_liquid * self.specific_heat_liquid

    @property
    def volumetric_heat_capacity_solid(self) -> float:
        """
        Calculate the volumetric heat capacity for the solid phase.
        Formula: volumetric_heat_capacity = density * specific_heat
        """
        return self.density_solid * self.specific_heat_solid

    @property
    def volumetric_latent_heat(self) -> float:
        """
        Calculate the volumetric latent heat of fusion.
        Formula: volumetric_latent_heat = density * specific_latent_heat
        """
        return self.density_liquid * self.specific_latent_heat

    @property
    def thermal_diffusivity_solid(self) -> float:
        """
        Calculate the thermal diffusivity for the solid phase.
        Formula: thermal_diffusivity = thermal_conductivity / volumetric_heat_capacity
        """
        return self.thermal_conductivity_solid / self.volumetric_heat_capacity_solid

    @property
    def thermal_diffusivity_liquid(self) -> float:
        """
        Calculate the thermal diffusivity for the liquid phase.
        Formula: thermal_diffusivity = thermal_conductivity / volumetric_heat_capacity
        """
        return self.thermal_conductivity_liquid / self.volumetric_heat_capacity_liquid

    def __str__(self):
        s = (
            f"Material properties:\n"
            f"  Phase Transition Temperature: {self.u_pt} K\n"
            f"  Specific Heat (Liquid): {self.specific_heat_liquid} J/(kg⋅K)\n"
            f"  Specific Heat (Solid): {self.specific_heat_solid} J/(kg⋅K)\n"
            f"  Density (Liquid): {self.density_liquid} kg/m^3\n"
            f"  Density (Solid): {self.density_solid} kg/m^3\n"
            f"  Volumetric Heat Capacity (Liquid): {self.volumetric_heat_capacity_liquid:.2E} J/(m^3⋅K)\n"
            f"  Volumetric Heat Capacity (Solid): {self.volumetric_heat_capacity_solid:.2E} J/(m^3⋅K)\n"
            f"  Volumetric Latent Heat of Fusion: {self.volumetric_latent_heat:.2E} J/m^3\n"
            f"  Thermal Conductivity (Liquid): {self.thermal_conductivity_liquid} W/(m⋅K)\n"
            f"  Thermal Conductivity (Solid): {self.thermal_conductivity_solid} W/(m⋅K)\n"
            f"  Thermal Diffusivity (Liquid): {self.thermal_diffusivity_liquid:.2E} m^2/s\n"
            f"  Thermal Diffusivity (Solid): {self.thermal_diffusivity_solid:.2E} m^2/s\n"
        )
        return s
