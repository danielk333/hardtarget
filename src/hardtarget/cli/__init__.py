#This is needed so that the registration is performed
from . import convert
from . import analyze

#Then expose the main after registration
from . commands import main
