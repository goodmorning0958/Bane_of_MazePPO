import struct
import math

def ConvertRobotAction(Action: list[float], done) -> list[list[int]]:
    LWP3Packet = []
    VBase = 50 # Base velocity

    dir_, step_size = Action[0], Action[1]
    if abs(dir_) < 1e-5:                      
        VLeft = VBase
        VRight = VBase
    elif dir_ > 0:
        VLeft = VBase
        VRight = int(VBase * (1.0 - 2.0 * (min(1.0, dir_/(math.pi/4)))))
    elif dir_ < 0:
        VLeft = int(VBase * (1.0 - 2.0 * (min(1.0, abs(dir_)/(math.pi/4)))))
        VRight = VBase

    if VLeft < 0:
        VLeft = int(round((~VLeft & 255) + 1))
    if VRight < 0:
        VRight = int(round((~VRight & 255) + 1))

    # calculate energy penalty from velocities
    Penalty = (VLeft**2 + VRight**2) / (2 * VBase**2) * float(step_size)
    step_bytes = list(struct.pack('<i', int(round(step_size))))
  
    Payload = (VLeft - 127.5) / 127.5, (VRight - 127.5) / 127.5, step_bytes

    LWP3Packet.append(Payload)

    return LWP3Packet, Penalty




