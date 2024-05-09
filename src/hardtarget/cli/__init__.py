# This is needed so that the registration is performed
from . import cmd_download
from . import cmd_convert
from . import cmd_inspect
from . import cmd_check
from . import cmd_analyze


# Plotting support is conditional 
from hardtarget import plotting
if plotting is not None: 
    from . import cmd_plot

# Then expose the main after registration
from . commands import main
