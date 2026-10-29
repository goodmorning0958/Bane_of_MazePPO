"""Self-contained motor + gearbox physics. Public API: reset() and step() only."""

import numpy as np
import mujoco


def reset(model, data):
    """Reset the car, always domain-randomize it, and return ``(data, params)``.

    Motor: maxon DCX22L01EBKL570 reference, 7.4 V supply.
    Gearbox: fixed 10:1 reduction, 85% nominal efficiency. Small motor,
    electrical, drivetrain, contact and timing variations are sampled once per
    episode. The chassis geometry/mass/inertia and gear ratio never change.
    """
    rng = np.random.default_rng()
    params = {
        # Published reference-motor values, converted to SI.
        "motor_name": "maxon DCX22L01EBKL570 reference",
        "kt": 0.0229,                 # Nm/A; Ke has the same value in SI
        "ke": 0.0229,                 # V s/rad
        "resistance": 1.84,           # ohm
        "inductance": 0.000192,       # H
        "rotor_inertia": 0.0000009,   # kg m^2
        "no_load_current": 0.020,     # A
        "no_load_rpm": 4980.0,
        "continuous_current": 1.3,    # A; chosen driver regulation limit
        "reference_temperature_c": 25.0,
        "max_winding_temperature_c": 100.0,
        "copper_temperature_coefficient": 0.00393,
        "winding_to_case_k_per_w": 4.57,
        "case_to_air_k_per_w": 13.6,
        "winding_time_s": 21.4,
        "motor_time_s": 646.0,
        "voltage": 7.4,
        "battery_resistance": 0.060,
        "driver_resistance": 0.200,
        "lead_resistance": 0.020,
        "gear_ratio": 10.0,           # motor rad / wheel rad
        "gearbox_efficiency": 0.85,
        "zero_mode": "brake",
        "ambient_c": 25.0,
        "control_dt": 0.020,          # 50 Hz
        "command_delay": 0.010,
        "pwm_levels": 255,
        "tire_friction": 0.70,
        "torsional_friction": 0.001,
        "rolling_friction": 0.0008,
        "contact_time": 0.010,
        "bearing_damping": 0.000001,
        "bearing_friction": 0.00015,
    }
    # Mandatory, deliberately small per-episode sim-to-real variation.
    motor_scale = float(rng.uniform(0.985, 1.015))
    params["kt"] *= motor_scale
    params["ke"] *= motor_scale
    params["motor_gains"] = rng.uniform(0.975, 1.025, size=4)
    params["resistance"] *= float(rng.uniform(0.97, 1.03))
    params["voltage"] = float(rng.uniform(7.25, 7.55))
    params["battery_resistance"] *= float(rng.uniform(0.90, 1.10))
    params["gearbox_efficiency"] = float(rng.uniform(0.82, 0.88))
    params["tire_friction"] = float(rng.uniform(0.65, 0.75))
    params["rolling_friction"] *= float(rng.uniform(0.85, 1.15))
    params["bearing_damping"] *= float(rng.uniform(0.85, 1.15))
    params["bearing_friction"] *= float(rng.uniform(0.85, 1.15))
    params["contact_time"] = float(rng.uniform(0.009, 0.011))
    params["command_delay"] = float(rng.integers(8, 13)) * 0.001

    wheel_joints = ("fl_wheel", "fr_wheel", "bl_wheel", "br_wheel")
    motors = ("flmotor", "frmotor", "blmotor", "brmotor")
    contacts = ("fl_ground", "fr_ground", "bl_ground", "br_ground")
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in wheel_joints], dtype=int)
    actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in motors], dtype=int)
    pair_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, n)
                         for n in contacts], dtype=int)
    if np.any(joint_ids < 0) or np.any(actuator_ids < 0) or np.any(pair_ids < 0):
        raise ValueError("model.xml is missing required wheel, motor or contact names")
    if model.nu != 4 or model.nuserdata < 1072:
        raise ValueError("Use the supplied model.xml (four actuators, 1072 userdata values)")
    if not np.isclose(model.opt.timestep, 0.001):
        raise ValueError("The hard-coded controller delay requires a 0.001 s physics timestep")

    dofs = model.jnt_dofadr[joint_ids]
    motor_loss = params["kt"] * params["no_load_current"]
    dry_motor_loss = 0.5 * motor_loss
    no_load_rad_s = params["no_load_rpm"] * 2 * np.pi / 60
    viscous_motor_loss = 0.5 * motor_loss / no_load_rad_s
    ratio = params["gear_ratio"]
    efficiency = params["gearbox_efficiency"]

    mujoco.mj_resetData(model, data)
    model.dof_armature[dofs] = params["rotor_inertia"] * ratio**2
    model.dof_frictionloss[dofs] = dry_motor_loss * ratio * efficiency + params["bearing_friction"]
    model.dof_damping[dofs] = viscous_motor_loss * ratio**2 * efficiency + params["bearing_damping"]
    model.pair_friction[pair_ids] = [params["tire_friction"], params["tire_friction"],
                                     params["torsional_friction"],
                                     params["rolling_friction"], params["rolling_friction"]]
    model.pair_solref[pair_ids] = [params["contact_time"], 1.0]
    model.actuator_ctrllimited[actuator_ids] = False

    data.userdata[:] = 0.0
    data.userdata[6:10] = params["ambient_c"]
    data.userdata[10:14] = params["ambient_c"]
    data.userdata[15] = params["voltage"]
    data.userdata[41] = 20            
    data.userdata[42] = round(params["command_delay"] / model.opt.timestep)
    data.userdata[43] = 1     
    mujoco.mj_forward(model, data)
    return data, params


def step(model, data, action, params):
    """Advance exactly one 1 ms physics step.

    ``action`` is ``[left_pwm, right_pwm]`` in [-1, 1]. The returned movement is
    ``[world_dx_m, world_dy_m, wrapped_dyaw_rad]``. Call reset() before step().
    """
    action = np.asarray(action, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("action must contain two finite values [left_pwm, right_pwm]")
    if data.userdata[43] != 1:
        raise ValueError("Call reset() before step()")

    dt = float(model.opt.timestep)
    wheel_names = ("fl_wheel", "fr_wheel", "bl_wheel", "br_wheel")
    motor_names = ("flmotor", "frmotor", "blmotor", "brmotor")
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in wheel_names], dtype=int)
    dofs = model.jnt_dofadr[joint_ids]
    actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in motor_names], dtype=int)
    u = data.userdata
    tick = int(u[14])

    # 50 Hz sample-and-hold, 8-bit PWM quantization, then 10 ms transport delay.
    if tick % int(u[41]) == 0:
        clipped = np.clip(action, -1.0, 1.0)
        u[27:29] = np.rint(clipped * params["pwm_levels"]) / params["pwm_levels"]
    ring = u[48:1072].reshape(512, 2)
    ring[tick % 512] = u[27:29]
    delayed = ring[(tick - int(u[42])) % 512]
    u[0:2] = delayed
    commands = delayed[[0, 1, 0, 1]]

    wheel_speed = data.qvel[dofs].copy()
    motor_speed = params["gear_ratio"] * wheel_speed
    winding_temp = u[6:10].copy()
    winding_r = (params["resistance"] *
                 (1 + params["copper_temperature_coefficient"] *
                  (winding_temp - params["reference_temperature_c"])))
    total_r = winding_r + params["driver_resistance"] + params["lead_resistance"]
    emf = params["ke"] * motor_speed
    old_current = u[2:6].copy()

    # Eight iterations solve the shared battery sag caused by all four motors.
    bus = params["voltage"]
    mean_current = np.zeros(4)
    mean_square = np.zeros(4)
    new_current = old_current.copy()
    applied_voltage = np.zeros(4)
    for _ in range(8):
        requested_voltage = commands * bus
        applied_voltage = requested_voltage.copy()
        a = np.exp(-total_r * dt / params["inductance"])
        b = -np.expm1(-total_r * dt / params["inductance"])
        predicted = old_current * a + ((applied_voltage - emf) / total_r) * b
        driven = commands != 0
        above = driven & (predicted > params["continuous_current"])
        below = driven & (predicted < -params["continuous_current"])
        applied_voltage[above] = (emf[above] + total_r[above] *
                                  (params["continuous_current"] - old_current[above] * a[above]) /
                                  b[above])
        applied_voltage[below] = (emf[below] + total_r[below] *
                                  (-params["continuous_current"] - old_current[below] * a[below]) /
                                  b[below])
        applied_voltage = np.clip(applied_voltage, -bus, bus)
        target = (applied_voltage - emf) / total_r
        z = total_r * dt / params["inductance"]
        decay = np.exp(-z)
        phi = -np.expm1(-z) / z
        phi2 = -np.expm1(-2 * z) / (2 * z)
        difference = old_current - target
        new_current = target + difference * decay
        mean_current = target + difference * phi
        mean_square = np.maximum(0.0, target**2 + 2 * target * difference * phi +
                                 difference**2 * phi2)
        if params["zero_mode"] == "coast":
            coast = commands == 0
            new_current[coast] = 0.0
            mean_current[coast] = 0.0
            mean_square[coast] = 0.0
        battery_current = float(np.sum(applied_voltage * mean_current) / max(bus, 1e-9))
        bus = max(0.1, params["voltage"] - params["battery_resistance"] * battery_current)

    motor_torque = params["kt"] * params["motor_gains"] * mean_current
    motor_power = motor_torque * motor_speed
    efficiency = params["gearbox_efficiency"]
    ratio = params["gear_ratio"]
    wheel_torque = np.where(motor_power >= 0,
                            motor_torque * ratio * efficiency,
                            motor_torque * ratio / efficiency)
    wheel_power = wheel_torque * wheel_speed
    gearbox_loss = np.maximum(0.0, np.where(motor_power >= 0,
                                            motor_power - wheel_power,
                                            -wheel_power + motor_power))

    # Two-node winding/case temperature estimate and telemetry accumulation.
    copper_heat = winding_r * mean_square
    winding_to_case = ((u[6:10] - u[10:14]) /
                       params["winding_to_case_k_per_w"])
    winding_capacity = params["winding_time_s"] / params["winding_to_case_k_per_w"]
    case_capacity = params["motor_time_s"] / params["case_to_air_k_per_w"]
    u[6:10] += dt * (copper_heat - winding_to_case) / winding_capacity
    u[10:14] += dt * (winding_to_case -
                       (u[10:14] - params["ambient_c"]) /
                       params["case_to_air_k_per_w"]) / case_capacity
    source_power = params["voltage"] * battery_current
    u[17] += max(source_power, 0.0) * dt       # gross energy drawn, J
    u[18] += max(-source_power, 0.0) * dt      # returned energy, J
    u[19] += float(np.sum(copper_heat)) * dt
    u[44] += float(np.sum(gearbox_loss)) * dt
    u[45] = float(np.sum(gearbox_loss))
    u[2:6] = new_current
    u[15] = bus
    u[16] = battery_current
    u[23:27] = wheel_torque

    x0, y0 = data.qpos[:2].copy()
    qw, qx, qy, qz = data.qpos[3:7]
    yaw0 = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy*qy + qz*qz))
    data.ctrl[actuator_ids] = wheel_torque
    mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError("non-finite MuJoCo state")
    qw, qx, qy, qz = data.qpos[3:7]
    yaw1 = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy*qy + qz*qz))
    dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))
    u[14] += 1
    return np.array([data.qpos[0] - x0, data.qpos[1] - y0, dyaw]), data