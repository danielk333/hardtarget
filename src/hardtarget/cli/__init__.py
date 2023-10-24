#This is needed so that the registration is performed
from . import convert
from . import analyze
from . import drfinfo

#Then expose the main after registration
from . commands import main
