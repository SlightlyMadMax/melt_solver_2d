from pydantic import BaseModel, Field, validator


class ThermalParameters(BaseModel):
    u_pt: float = Field(..., gt=0.0, description="Phase transition temperature [K].")
    u_ref: float = Field(..., gte=0.0, description="Reference temperature [K].")
    specific_heat_liquid: float = Field(
        ...,
        gt=0.0,
        description="Specific heat capacity of the liquid phase [J/(kg⋅K)].",
    )
    specific_heat_solid: float = Field(
        ..., gt=0.0, description="Specific heat capacity of the solid phase [J/(kg⋅K)]."
    )
    specific_latent_heat_solid: float = Field(
        ...,
        gt=0.0,
        description="Specific latent heat of fusion of the solid phase [J/kg].",
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
    delta: float = Field(
        None, gt=0.0, description="Default smoothing parameter (delta)."
    )

    @property
    def u_pt_ref(self) -> float:
        """
        Calculate the deviation of phase transition temperature from the reference temperature.
        """
        return self.u_pt - self.u_ref

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
    def volumetric_latent_heat_solid(self) -> float:
        """
        Calculate the volumetric latent heat of fusion for the solid phase
        (given its density is equal to the density of the liquid phase when melting/freezing).
        Formula: volumetric_latent_heat = density * specific_latent_heat
        """
        return self.density_liquid * self.specific_latent_heat_solid

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
            f"Heat Transfer Parameters:\n"
            f"  Phase Transition Temperature: {self.u_pt} K\n"
            f"  Reference Temperature: {self.u_ref} K\n"
            f"  Specific Heat (Liquid): {self.specific_heat_liquid} J/(kg⋅K)\n"
            f"  Specific Heat (Solid): {self.specific_heat_solid} J/(kg⋅K)\n"
            f"  Density (Liquid): {self.density_liquid} kg/m^3\n"
            f"  Density (Solid): {self.density_solid} kg/m^3\n"
            f"  Volumetric Heat Capacity (Liquid): {self.volumetric_heat_capacity_liquid:.2E} J/(m^3⋅K)\n"
            f"  Volumetric Heat Capacity (Solid): {self.volumetric_heat_capacity_solid:.2E} J/(m^3⋅K)\n"
            f"  Volumetric Latent Heat of Fusion (Solid): {self.volumetric_latent_heat_solid:.2E} J/m^3\n"
            f"  Thermal Conductivity (Liquid): {self.thermal_conductivity_liquid} W/(m⋅K)\n"
            f"  Thermal Conductivity (Solid): {self.thermal_conductivity_solid} W/(m⋅K)\n"
            f"  Thermal Diffusivity (Liquid): {self.thermal_diffusivity_liquid:.2E} m^2/s\n"
            f"  Thermal Diffusivity (Solid): {self.thermal_diffusivity_solid:.2E} m^2/s\n"
            f"  Default Smoothing Parameter (Delta): {self.delta or "-"}\n"
        )
        return s
