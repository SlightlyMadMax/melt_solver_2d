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
        dynamic_viscosity=1.7888e-3,
        volumetric_thermal_exp=-6.733353e-05,
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
