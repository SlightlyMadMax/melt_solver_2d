from .douglas_rachford import DRNavierStokesScheme
from .peaceman_rachford import PRNavierStokesScheme
from .explicit import ExplicitNavierStokesSolver
from .loc_one_dim import LODNavierStokesScheme
from .vabishchevich import VabishchevichScheme
from .registry import VorticitySolverName, VorticitySolverRegistry
