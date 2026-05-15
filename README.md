# 2D Stefan Problem Solver with Convection

Numerical solver for 2D melting/solidification problems with natural convection.
Combines the stream function–vorticity formulation of the incompressible Navier–Stokes
equations with the effective heat capacity method for phase-change heat transfer.

## Physical model

The velocity field is recovered from the stream function:
$v_x = \partial\psi/\partial y$, $v_y = -\partial\psi/\partial x$.

All equations are written in dimensionless form using the characteristic length $L$,
the buoyancy velocity $V = \sqrt{g\lvert\beta\rvert\Delta T L}$, and the dimensionless
temperature $\theta = (T - T_\text{ref})/\Delta T$.

- **Vorticity transport** (with Brinkman / fictitious-domain penalty term $S$ for the solid):

$$\frac{\partial \omega}{\partial t} + \frac{\partial}{\partial x}\left(\omega \frac{\partial \psi}{\partial y}\right) - \frac{\partial}{\partial y}\left(\omega \frac{\partial \psi}{\partial x}\right) = \frac{1}{Re}\nabla^2\omega + \frac{Gr}{Re^2}\frac{\partial\theta}{\partial x} - \nabla \cdot (S \nabla\psi)$$

- **Stream function Poisson equation**:

$$\nabla^2\psi = -\omega$$

- **Energy equation** (effective heat capacity method):

$$c_\text{eff}(\theta)\left(\frac{\partial\theta}{\partial t} + \frac{\partial(v_x \theta)}{\partial x} + \frac{\partial(v_y \theta)}{\partial y}\right) = \frac{1}{Pe}\nabla\cdot\left(k_\text{eff}(\theta)\nabla\theta\right)$$

where the effective heat capacity absorbs the latent heat release:

$$c_\text{eff}(\theta) = \frac{\tilde{c}(\theta)}{c_\text{ref}} + \frac{\delta_\Delta(\theta - \theta_m)}{Ste}$$

$\delta_\Delta$ is a smooth approximation of the Dirac delta that smears the
solid–liquid interface over a mushy zone of half-width $\Delta$ (in dimensionless
temperature units). The penalty $S$ is a smooth function of $\theta$ that drives the
velocity to zero in the solid phase.

## Project structure

```
src/
├── main.py                          # Example entry point
├── examples/
│   ├── stefan/                      # Pure conduction Stefan problem
│   ├── gallium/                     # Gallium melting with convection
│   ├── octadecane/                  # n-Octadecane in a differentially heated cavity
│   ├── water_convection/            # Natural convection in liquid water
│   ├── water_freezing/              # Water freezing with convection
│   ├── horizontal_layer/            # Horizontal layer melting
│   ├── icicle/                      # Icicle growth
│   ├── crevasse/                    # Crevasse melting
│   └── air/                         # Air convection reference case
├── parameters/
│   ├── config.py                    # ExperimentConfig (Pydantic model, JSON-loadable)
│   └── material_properties.py       # MaterialProperties (Pydantic model)
├── core/
│   ├── geometry.py                  # DomainGeometry
│   ├── boundary_conditions.py       # BoundaryCondition, BoundaryConditions
│   ├── runner.py                    # SimulationState, ExperimentRunner
│   └── solvers/
│       ├── tridiagonal_solver.py    # Numba-JIT Thomas algorithm
│       └── mixins/adi.py            # ADIMixin (shared ADI sweep infrastructure)
├── convective_operators/
│   ├── sf_based.py                  # Velocity-field convective operators
│   └── vorticity_based.py           # Jacobian (Arakawa-type) convective operator
├── fluid_dynamics/
│   ├── utils.py                     # Vorticity/velocity helpers, BC mixin
│   └── solvers/
│       ├── bc_correction_solver_factory.py       # BCCorrectionNVSolver (coupled ψ–ω)
│       ├── stream_function_solvers/
│       │   ├── amg.py               # Algebraic multigrid (PyAMG)
│       │   ├── cg.py                # Conjugate gradient (CPU)
│       │   ├── cg_gpu.py            # Conjugate gradient (GPU / CuPy)
│       │   ├── sor.py               # Successive over-relaxation (classical Poisson only)
│       │   └── matrix_sweep.py      # Direct tridiagonal sweep (classical Poisson only)
│       └── vorticity_solvers/
│           ├── peaceman_rachford.py  # Peaceman–Rachford ADI
│           ├── douglas_rachford.py   # Douglas–Rachford ADI
│           ├── loc_one_dim.py        # Locally one-dimensional (LOD)
│           ├── vabishchevich.py      # Vabishchevich splitting
│           ├── explicit.py           # Explicit (forward Euler)
│           └── vab_fully_implicit.py # Vabishchevich fully implicit
├── heat_transfer/
│   ├── coefficient_smoothing/
│   │   ├── coefficients.py          # Step/delta function schemes (erf, tanh, linear…)
│   │   └── mushy_zone.py            # Adaptive Δ estimation from grid
│   └── solvers/heat_transfer_solvers/
│       ├── peaceman_rachford.py      # PR ADI heat solver
│       ├── douglas_rachford.py       # DR ADI heat solver
│       ├── loc_one_dim.py            # LOD heat solver
│       ├── fully_implicit.py         # Fully implicit heat solver
│       └── explicit.py              # Explicit heat solver
└── utils/
    ├── boundary_conditions.py        # Dirichlet/Neumann BC factories
    └── nusselt.py                    # Nusselt number calculation
```

## Examples

Ready-to-run simulations are provided in `src/examples/`, each in its own subdirectory
with a `config.json` and a `run.py` entry point:

| Directory | Problem |
|---|---|
| `stefan/` | Pure conduction Stefan problem (no flow) |
| `gallium/` | Melting of gallium with natural convection |
| `octadecane/` | Melting of n-octadecane in a differentially heated cavity |
| `water_convection/` | Natural convection in liquid water |
| `water_freezing/` | Freezing of water with convection |
| `horizontal_layer/` | Horizontal layer melting |
| `icicle/` | Icicle growth |
| `crevasse/` | Crevasse melting |
| `air/` | Air convection reference case |

Each `run.py` shows the full solver setup for that material and geometry and can be
used as a template for new cases.

## Installation

Requires Python 3.11 or 3.12.

```bash
pip install -r requirements.txt
# or, with Poetry:
poetry install
```

> **GPU support** requires CuPy with a CUDA 12.x runtime.

## Configuration

Experiments are configured via JSON files loaded into `ExperimentConfig`:

```python
from src.parameters.config import ExperimentConfig

cfg = ExperimentConfig.load_from_file("parameter_sets/my_case/config.json")
```

A config file specifies:

| Field | Description |
|---|---|
| `geometry` | `width`, `height`, `end_time`, `n_x`, `n_y`, `n_t` |
| `u_ref` | Reference temperature [K] |
| `delta_u` | Characteristic temperature difference [K] |
| `l` | Characteristic length [m] |
| `delta` | Half-width of mushy zone for heat transfer [K] (optional; auto-estimated if absent) |
| `delta_flow` | Half-width of mushy zone for flow [K] (optional) |
| `epsilon` | Penalty parameter for fictitious domain method |
| `material_props` | See `MaterialProperties` below |

`MaterialProperties` fields: `u_pt`, `specific_heat_liquid`, `specific_heat_solid`,
`specific_latent_heat`, `density_liquid`, `density_solid`,
`thermal_conductivity_liquid`, `thermal_conductivity_solid`,
`dynamic_viscosity`, `volumetric_thermal_exp`, `density_poly_coeffs` (optional).

`ExperimentConfig` computes dimensionless numbers (`Re`, `Gr`, `Ra`, `Pr`, `Pe`, `Ste`)
from the material properties and characteristic scales, and exposes scaled grid steps.

## Usage

### Running a simulation

```python
from src.core.boundary_conditions import BoundaryConditions
from src.core.runner import SimulationState, ExperimentRunner
from src.fluid_dynamics.init_values import initialize_stream_function, initialize_vorticity, initialize_velocity
from src.fluid_dynamics.solvers import VorticitySolverName, StreamFunctionSolverName
from src.fluid_dynamics.solvers.bc_correction_solver_factory import BCCorrectionNVSolver
from src.fluid_dynamics.solvers.vorticity_solvers.base_solver import PenaltyTermForm
from src.heat_transfer.init_values import init_temperature, DomainShape
from src.heat_transfer.solvers import HeatTransferSolver, HeatTransferSolverName
from src.heat_transfer.solvers.heat_transfer_solvers.base_solver import KFaceMethod
from src.heat_transfer.coefficient_smoothing.coefficients import StepScheme, DeltaScheme
from src.convective_operators import ConvectiveTermForm
from src.parameters.config import ExperimentConfig
from src.utils.boundary_conditions import const_dirichlet_condition, const_neumann_condition

cfg = ExperimentConfig.load_from_file("parameter_sets/my_case/config.json")
geometry = cfg.geometry

u_bcs = BoundaryConditions(
    left=const_dirichlet_condition(geometry.n_y, value=1.0),
    right=const_dirichlet_condition(geometry.n_y, value=0.0),
    top=const_neumann_condition(geometry.n_x, value=0.0),
    bottom=const_neumann_condition(geometry.n_x, value=0.0),
)
sf_bcs = BoundaryConditions(
    left=const_dirichlet_condition(geometry.n_y, value=0.0),
    right=const_dirichlet_condition(geometry.n_y, value=0.0),
    top=const_dirichlet_condition(geometry.n_x, value=0.0),
    bottom=const_dirichlet_condition(geometry.n_x, value=0.0),
)

u  = init_temperature(cfg=cfg, bcs=u_bcs, shape=DomainShape.UNIFORM_SOLID, solid_temp=cfg.material_props.u_pt - 1.0)
sf = initialize_stream_function(geometry=geometry, bcs=sf_bcs)
w  = initialize_vorticity(geometry=geometry)
v_x, v_y = initialize_velocity(geometry=geometry)

heat_solver = HeatTransferSolver(
    cfg=cfg,
    bcs=u_bcs,
    solver_name=HeatTransferSolverName.PEACEMAN_RACHFORD,
    convective_term_form=ConvectiveTermForm.DEFERRED_CORRECTION,
    step_scheme=StepScheme.ERF,
    delta_scheme=DeltaScheme.GAUSS,
    k_face_method=KFaceMethod.FROM_TEMP,
    max_iters=1,
    tolerance=1e-6,
    urf=1.0,
)

navier_solver = BCCorrectionNVSolver(
    cfg=cfg,
    sf_bcs=sf_bcs,
    vorticity_solver_name=VorticitySolverName.PEACEMAN_RACHFORD,
    stream_function_solver_name=StreamFunctionSolverName.AMG,
    convective_term_form=ConvectiveTermForm.DIVERGENT_CENTRAL,
    penalty_term_form=PenaltyTermForm.LINEAR,
    vorticity_bc_order=2,
)

state = SimulationState(u=u, sf=sf, w=w, v_x=v_x, v_y=v_y)

runner = ExperimentRunner(
    cfg=cfg,
    state=state,
    heat_solver=heat_solver,
    navier_solver=navier_solver,
    checkpoints_dir="data/my_case",
    save_at={500, 1000, 2000},
)
runner.run()
```

### Resuming from a checkpoint

```python
runner = ExperimentRunner.from_checkpoint(
    checkpoint_path="data/my_case/checkpoint_1000.npz",
    cfg=cfg,
    heat_solver=heat_solver,
    navier_solver=navier_solver,
    checkpoints_dir="data/my_case",
    save_at={2000},
)
runner.run()
```

## Solver options

### Vorticity solvers (`VorticitySolverName`)

| Name | Scheme | Notes |
|---|---|---|
| `PEACEMAN_RACHFORD` | Peaceman–Rachford ADI | Recommended |
| `DOUGLAS_RACHFORD` | Douglas–Rachford ADI | Unconditionally stable |
| `LOC_ONE_DIM` | Locally one-dimensional | Full-step per direction |
| `EXPLICIT` | Forward Euler | Stable only for small time steps |

### Stream function solvers (`StreamFunctionSolverName`)

| Name | Method | Notes |
|---|---|---|
| `AMG` | Algebraic multigrid (PyAMG) | Fastest for large grids; recommended |
| `CG` | Conjugate gradient (SciPy) | Good fallback |
| `CG_GPU` | Conjugate gradient (CuPy) | Requires CUDA 12.x |
| `SOR` | Successive over-relaxation | Only for the classical Poisson equation $\nabla^2\psi = -\omega$ |
| `MATRIX_SWEEP` | Direct tridiagonal sweep | Only for the classical Poisson equation $\nabla^2\psi = -\omega$ |

### Heat transfer solvers (`HeatTransferSolverName`)

| Name | Scheme |
|---|---|
| `PEACEMAN_RACHFORD` | Peaceman–Rachford ADI |
| `DOUGLAS_RACHFORD` | Douglas–Rachford ADI |
| `LOC_ONE_DIM` | Locally one-dimensional |
| `FULLY_IMPLICIT` | Fully implicit |
| `EXPLICIT` | Forward Euler |

### Convective term forms (`ConvectiveTermForm`)

| Name | Description |
|---|---|
| `DIVERGENT_CENTRAL` | Central difference of $\partial(v_i\phi)/\partial x_i$ |
| `NON_DIVERGENT_CENTRAL` | Central difference of $v_i\,\partial\phi/\partial x_i$ |
| `SYMMETRIC` | Average of divergent and non-divergent |
| `UPWIND_NC` | Node-centered first-order upwind |
| `UPWIND_FC` | Face-centered first-order upwind |
| `DEFERRED_CORRECTION` | Upwind base + high-order correction (heat transfer only) |

### Step / delta function schemes

**Step** (`StepScheme`): `ERF`, `HYPER`, `LINEAR`, `CONST`, `JUMP`

**Delta** (`DeltaScheme`): `GAUSS`, `HYPER`, `PARABOLIC`, `BOX`

### Penalty term forms (`PenaltyTermForm`)

| Name | Formula |
|---|---|
| `JUMP` | $C \cdot \mathbf{1}_{u \le u_0}$ |
| `LINEAR` | $C \cdot \frac{1}{2}(1 - \tanh(\tilde{u}/\Delta))$ |
| `QUADRATIC` | $C \cdot (1 - f_l)^2$ |
| `KOZENY_CARMAN` | $C \cdot (1 - f_l)^2 / (f_l^3 + \varepsilon)$ |

### Thermal conductivity at faces (`KFaceMethod`)

`ARITHMETIC`, `HARMONIC`, `FROM_TEMP` (evaluates step function at the face temperature).

## Running tests

```bash
pytest tests/
```
