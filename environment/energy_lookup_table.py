import csv
import numpy as np

with open("environment/physics_calibration/calibration_moves.csv", newline="") as f:
    rows = list(csv.DictReader(f))

def curve(target, joules):
    groups = {}
    for r in rows:
        if float(r[target]) > 0:
            groups.setdefault(float(r[target]), []).append(float(r[joules]))
    x = [0.0] + sorted(groups)
    return x, [0.0] + [np.mean(groups[k]) for k in x[1:]]

turn_x, turn_j = curve("target_turn", "turn_energy_J")
step_x, step_j = curve("target_step_units", "drive_energy_J")

def move_joules(turn, step):
    return float(np.interp(abs(turn), turn_x, turn_j) + np.interp(step, step_x, step_j))