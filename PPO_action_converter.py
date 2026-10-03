import struct

def ConvertRobotAction(Action: list[float], done) -> list[list[int]]:
    """
    IN FORM [dir_, step_size]
    OUT FORM: [Length, HubID, CommandType, PortID, executionInfo, Sub-Command, LefPWM, RightPWM, EndState]

    Notes:
    V[varible] = velocity
    """
    LWP3Packet = []
    VBase = 50 # Base velocity

    dir_, step_size = Action[0], Action[1]
    if dir_ == 0:
        VLeft = VBase
        VRight = VBase
    elif dir_ > 0:
        VLeft = VBase
        VRight = int(VBase * (1.0 - 2.0 * (min(1.0, dir_/90))))
    elif dir_ < 0:
        VLeft = int(VBase * (1.0 - 2.0 * (min(1.0, abs(dir_)/90))))
        VRight = VBase

    # Convert to LEGO motor params (one byte)
    if VLeft < 0:
        VLeft = int(round((~VLeft & 255) + 1))
    if VRight < 0:
        VRight = int(round((~VRight & 255) + 1))

    # calculate energy penalty from velocities
    Penalty = (VLeft**2 + VRight**2) >> 14
    step_bytes = list(struct.pack('<i', int(round(step_size))))
    Payload = [Length, HubID, CommandType, PortID, ExecutionInfo, SubCommand] + step_bytes + [VLeft, VRight, EndState, Profile]

    # Other packet Info
    HubID = 0x00
    CommandType = 0x81   # Port Output Command
    PortID = 0x10        # Virtual Synchronized Port for Left+Right Motors
    ExecutionInfo = 0x10 # Request Command Feedback notification when done
    SubCommand = 0x08    # Output for Degrees, Synchronized
    EndState = 0x7F if done else 0x00 # Robot Path Has Been Completed
    Profile = 0x00       # No speed profile ramps
    Length = len(Payload) + 1

    LWP3Packet.append(Payload)

    return LWP3Packet, Penalty




