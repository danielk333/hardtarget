# This is needed so that the registration is performed
# Plotting support is conditional
from hardtarget import plotting

from . import cmd_analyse, cmd_check, cmd_inspect

if plotting is not None:
    from . import cmd_plot

# Then expose the main after registration
from .commands import main
