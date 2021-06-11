import os
#Quite hacky way of finding the modules path. Will only work on 
#unix like file systems, because of the '/'.
PATH = os.path.abspath(__file__)[:-len(__file__.split('/')[-1])]
