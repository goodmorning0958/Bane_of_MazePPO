import struct
import math
from energy_lookup_table import move_joules

ENERGY_PER_RADIAN = 0.5

def ConvertRobotAction(Action):
    LWP3Packet = []
    VBase = 50 # Base velocity

    dir_, step_size = Action[0], Action[1]

    Penalty = move_joules(dir_, step_size)

    return Penalty