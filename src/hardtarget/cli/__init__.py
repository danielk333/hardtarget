#This is needed so that the registration is performed
from . import convert
from . import analyze
from . import cudacheck
from . import info_drf
from . import info_gmf
from . import plot_drf
from . import plot_gmf

#Then expose the main after registration
from . commands import main
