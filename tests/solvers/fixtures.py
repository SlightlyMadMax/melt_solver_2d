from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties


def get_cfg() -> ExperimentConfig:
    geometry = DomainGeometry(
        width=1.0,
        height=1.0,
        end_time=100,
        n_x=100,
        n_y=100,
        n_t=100,
    )
    material_props = MaterialProperties(
        u_pt=273.15,
        specific_heat_liquid=4200,
        specific_heat_solid=2100,
        specific_latent_heat=3.33e5,
        density_liquid=1000,
        density_solid=1000,
        thermal_conductivity_liquid=0.59,
        thermal_conductivity_solid=2.21,
        kinematic_viscosity_coeffs=[
            0.000108963453,
            -9.28722151e-07,
            2.65889022e-09,
            -2.54761652e-12,
        ],
        volumetric_thermal_exp_coeffs=[7.68e-6],
    )
    cfg = ExperimentConfig(
        geometry=geometry,
        material_props=material_props,
        u_ref=273.15,
        delta_u=1.0,
        v=0.1,
        l=1.0,
        delta=None,
        epsilon=1e-6,
    )
    return cfg
