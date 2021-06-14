'''
Exposes a global PATH variable to find the path to where the 
module is installed.
'''
import os
#Quite hacky way of finding the modules path. Tested on both 
#Windows and Ubuntu so should work for most file systems
_PATH = os.path.abspath(__file__)[:-len(__file__.split('/')[-1])]
