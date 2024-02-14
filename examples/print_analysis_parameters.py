"""

MAKE DOC HERE
=============

todo

"""

from pprint import pprint
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
config = "/home/danielk/git/hard_target/examples/cfg/test.ini"

params = hardtarget.load_gmf_params(target, config)
pprint(params)
