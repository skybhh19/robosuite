from robosuite.environments.base import make

# Manipulation environments
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.environments.manipulation.tidy import Tidy
from robosuite.environments.manipulation.cleanup import CleanUp
from robosuite.environments.manipulation.cleanup_medium import CleanUpMediumSmallInitA, CleanUpMediumSmallInitA1, CleanUpMediumSmallInitAReal1, CleanUpMediumSmallInitB, CleanUpMediumSmallInitB1, CleanUpMediumSmallInitBReal1, CleanUpMediumSmallInitC, CleanUpMediumSmallInitC1, CleanUpMediumSmallInitCReal1, CleanUpMediumSmallInitD, CleanUpMediumSmallInitD1, CleanUpMediumSmallInitD2, CleanUpMediumSmallInitD3, CleanUpMediumSmallInitDa1, CleanUpMediumSmallInitDReal1, CleanUpMediumSmallInitDObjectTrain, CleanUpMediumSmallInitDObjectTest, CleanUpMediumLargeInit
from robosuite.environments.manipulation.kitchen import Kitchen

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.3.1"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
