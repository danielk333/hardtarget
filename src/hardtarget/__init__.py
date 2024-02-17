from .version import __version__

# Submodules and packages
from . import analysis
from . import simulation
from . import noise
from . import configuration

# Constants and singletons
from .gmf import GMF_GRID_LIBS, GMF_OPTIMIZE_LIBS
from .experiments import EXP_FILES

# Functions
from .experiments import load_expconfig, load_radar_code
from .drf_utils import load_hardtarget_drf
from .configuration import load_gmf_params
from .analysis import load_gmf_out, compute_gmf

# Logging
from .profiling import setup_loggers

# Plotting
try:
    from . import plotting
except ImportError:
    plotting = None

# Profiling
try:
    from . import profiling
    from .profiling import profile, profile_stop, get_profile, print_profile, profile_clear
except ImportError:
    profiling = None
