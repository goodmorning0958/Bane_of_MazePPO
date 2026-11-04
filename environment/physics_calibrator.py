import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import mujoco
from environment.physics import reset, step

turns = [round(0.02 * i, 2) for i in range(1, 158)]
steps = [round(0.02 * i, 2) for i in range(1, 76)]
trials = 10
seed = 42
pwm = 0.6
lead_s = 0.04
settle_s = 0.3
stop_s = 1.0
timeout_s = 8.0
out = Path("environment") / "physics_calibration"

_rng = np.random.default_rng(seed)
np.random.default_rng = lambda *args, **kwargs: _rng


def yaw_of(data):
    qw, qx, qy, qz = data.qpos[3:7]
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def still(data):
    return abs(float(data.qvel[5])) < 0.02 and math.hypot(float(data.qvel[0]), float(data.qvel[1])) < 0.002

def hold(model, data, params, seconds):
    for _ in range(int(round(seconds / model.opt.timestep))):
        step(model, data, [0.0, 0.0], params)


def spin(model, data, params, target):
    dt = float(model.opt.timestep)
    turned = 0.0
    yaw_prev = yaw_of(data)
    braking = False
    brake_ticks = 0
    still_ticks = 0
    for _ in range(int(round(timeout_s / dt))):
        if not braking and target - turned <= abs(float(data.qvel[5])) * lead_s:
            braking = True
        step(model, data, [0.0, 0.0] if braking else [-pwm, pwm], params)
        yaw = yaw_of(data)
        turned += math.atan2(math.sin(yaw - yaw_prev), math.cos(yaw - yaw_prev))
        yaw_prev = yaw
        if braking:
            brake_ticks += 1
            still_ticks = still_ticks + 1 if still(data) else 0
            if still_ticks * dt >= 0.05 or brake_ticks * dt >= stop_s:
                break
    return abs(turned)


def drive(model, data, params, target_m):
    dt = float(model.opt.timestep)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    braking = False
    brake_ticks = 0
    still_ticks = 0
    for _ in range(int(round(timeout_s / dt))):
        dist = math.hypot(float(data.qpos[0]) - x0, float(data.qpos[1]) - y0)
        speed = math.hypot(float(data.qvel[0]), float(data.qvel[1]))
        if not braking and target_m - dist <= speed * lead_s:
            braking = True
        step(model, data, [0.0, 0.0] if braking else [pwm, pwm], params)
        if braking:
            brake_ticks += 1
            still_ticks = still_ticks + 1 if still(data) else 0
            if still_ticks * dt >= 0.05 or brake_ticks * dt >= stop_s:
                break
    return math.hypot(float(data.qpos[0]) - x0, float(data.qpos[1]) - y0)


def move(model, data, turn, step_units):
    data, params = reset(model, data)
    hold(model, data, params, settle_s)
    u = data.userdata

    e0 = u[[17, 18, 19, 44]].copy()
    t0 = int(u[14])

    turned = spin(model, data, params, turn) if turn > 0 else 0.0
    driven = drive(model, data, params, step_units * 0.5) if step_units > 0 else 0.0

    t1 = int(u[14])
    e1 = float(u[17])
    e = u[[17, 18, 19, 44]] - e0

    return dict(target_turn = turn, target_step_units = step_units, turn_rad = turned, distance_m = driven,
                distance_units = driven / 0.5, energy_J = float(e[0]), turn_energy_J = e1 - float(e0[0]),
                drive_energy_J = float(u[17]) - e1, returned_J = float(e[1]), copper_J = float(e[2]), gearbox_J = float(e[3]),
                turn_time_s = (t1 - t0) * float(model.opt.timestep), time_s = (int(u[14]) - t0) * float(model.opt.timestep),
                voltage = params["voltage"], gearbox_efficiency = params["gearbox_efficiency"], tire_friction = params["tire_friction"])


def fit(rows, columns):
    x = np.array([[c(r) for c in columns] for r in rows], dtype = float)
    y = np.array([r["energy_J"] for r in rows], dtype = float)

    coef, *_ = np.linalg.lstsq(x, y, rcond = None)
    pred = x @ coef

    r2 = 1 - float(np.sum((y - pred) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    return [float(v) for v in coef], r2, rmse

if __name__ == "__main__":
    out.mkdir(parents = True, exist_ok = True)

    model = mujoco.MjModel.from_xml_path("environment/model.xml")
    data = mujoco.MjData(model)
    combos = [(t, 0.0) for t in turns] + [(0.0, s) for s in steps]
    total = len(combos) * trials

    rows = []
    start = time.perf_counter()

    for trial in range(trials):
        for turn, step_units in combos:
            r = move(model, data, turn, step_units)
            r["trial"] = trial
            rows.append(r)

    with open(out / "calibration_moves.csv", "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    d = lambda r: r["distance_units"]
    a = lambda r: r["turn_rad"]
    one = lambda r: 1.0
    d_on = lambda r: 1.0 if r["target_step_units"] > 0 else 0.0
    a_on = lambda r: 1.0 if r["target_turn"] > 0 else 0.0

    (a1, b1), r2_1, e_1 = fit(rows, [d, a])
    (a2, b2, c2), r2_2, e_2 = fit(rows, [d, a, one])
    (a3, b3, cd3, ct3), r2_3, e_3 = fit(rows, [d, a, d_on, a_on])

    summary = {
        "unit_m": 0.5, "pwm": pwm, "trials": trials, "moves": len(rows), "seed": seed,
        "model_1": {"formula": "E = a*distance + b*turn", "a_J_per_unit": a1, "b_J_per_rad": b1,
                    "energy_per_radian": b1 / a1, "r2": r2_1, "rmse_J": e_1},
        "model_2": {"formula": "E = a*distance + b*turn + c", "a_J_per_unit": a2, "b_J_per_rad": b2, "c_J_per_move": c2,
                    "energy_per_radian": b2 / a2, "energy_per_move": c2 / a2, "r2": r2_2, "rmse_J": e_2},
        "model_3": {"formula": "E = a*distance + b*turn + c_drive*[moved] + c_turn*[turned]", "a_J_per_unit": a3,
                    "b_J_per_rad": b3, "c_drive_J": cd3, "c_turn_J": ct3, "r2": r2_3, "rmse_J": e_3},
        "current_rule": {"energy_per_radian": 0.5},
        "minutes": (time.perf_counter() - start) / 60,
    }
    
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent = 2), encoding = "utf-8")