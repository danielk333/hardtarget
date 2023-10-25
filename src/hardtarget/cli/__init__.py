#This is needed so that the registration is performed
from . import convert
from . import analyze
from . import drfinfo
from . import cudacheck

#Then expose the main after registration
from . commands import main
