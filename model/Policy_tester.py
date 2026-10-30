import csv
import math
import time
import numpy as np
import torch
from environment.training_example_generator import get_new_training_example
from environment.A_star_search import Normal_A_star_search
from environment.RunIntoWall import RunIntoWall
import random
from pathlib import Path

from model.actor_critic import (
    ActorCritic,
    RAY_SCALE,
)
from environment.maze_utils import _build_bsp_tree
from environment.energy_lookup_table import move_joules

# Flags
ROLLOUTS = 3000
SEED = 42
CURRICULUM_STAGES = [30, 40, 50, 51, 52, 54, 55, 56, 53]
STALL_WINDOW = 20                  
STALL_MIN_DISPLACEMENT = 2.0     
STALL_END_PENALTY = 1.0

# main
def StartAgent(test, NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType, VARtargetMaxTurn, agnt, VARFOV, VARRayDistance, *, seed=SEED, rollouts=None, output_dir=None):
    rollouts = ROLLOUTS if rollouts is None else int(rollouts)

    directory = Path(output_dir) if output_dir is not None else Path("runs") / f"agent_{agnt}_seed_{seed}_{time.time_ns()}"
    directory.mkdir(parents=True, exist_ok=True)
    test = "eval_" + test
    success_rate = 0.0
    episode_count = 0
    past = 0
    not_past = 0
    energy_ramp_rate = 1.0
    start_time = time.perf_counter()

    # classes
    network = ActorCritic(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, VARFOV, NNinputs)
    network.load_state_dict(torch.load(directory / f"Policy_{agnt}.pt", weights_only=False)["model_state_dict"])
    network.set_max_step(0.0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    headers = {
        "EpisodeData": ["episode", "reward", "steps", "success", "success_rate", "raw_energy", "regions", "status",
                        "penalty_sum", "path_length", "turn_radians", "lidar_seconds", "inference_seconds", "rays_cast", "termination",
                        "maze_size", "difficulty_x", "stage", "optimal_length", "path_efficiency", "accuracy", "energy_per_length",
                        "sensing_units", "max_step_cap", "revisit_steps", "elapsed_seconds", "physics_energy_J"],
    }

    for name, header in headers.items():
        with open(directory / f"{name}_{test}{agnt}.csv", "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    for _rollout in range(episode_count, rollouts):
        x = random.choice(CURRICULUM_STAGES)
        
        # get training example
        start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd)
        robot_pos = np.array(start_point, dtype=np.float32)
        end_point = np.array(end_point, dtype=np.float32)
        walls_np, BSP = _build_bsp_tree(walls)
        robot_pos = robot_pos + 0.5
        end_point = end_point + 0.5
        episode_x, episode_stage = x, CURRICULUM_STAGES.index(x)

        # Initializations
        angle = 0.0
        A_star_path = Normal_A_star_search(np.array(pixel_grid), [int(v) for v in start_point], [int(v) for v in np.floor(end_point)])
        A_star_points = np.asarray(A_star_path, dtype=float)
        optimal_length = float(np.linalg.norm(np.diff(A_star_points, axis=0), axis=1).sum()) if len(A_star_points) > 1 else 0.0
        total_turn_delta = 0.0
        RegionsExplored = set()
        PixelVisits = np.zeros(np.asarray(pixel_grid).shape)    # how often each pixel has been entered this episode
        revisit_steps = 0
        sensing_units = 0.0
        ray_range = float(maze_size if VARRayDistance is None else VARRayDistance)
        stalled = False

        start_region = goal_region = None
        if grid_distance > 0:
            start_region = tuple((np.floor(robot_pos / grid_distance) * grid_distance).tolist())
            goal_region = tuple((np.floor(end_point / grid_distance) * grid_distance).tolist())
            RegionsExplored.add(start_region)

        network.EnergyUsed = 0.0
        network.last_action = [0.0, 0.0]
        robot_path = [robot_pos.copy()]
        episode_reward = 0.0
        episode_length = 0
        lidar_seconds_this_rollout = 0.0
        inference_seconds = 0.0
        rays_this_rollout = 0
        penalty_sum = 0.0
        path_length = 0.0
        physics_energy = 0.0
        maze_solved = False
        termination = "time_limit"

        for step in range(network.MaxSteps):
            done = False
            distance_finish = float(np.linalg.norm(end_point - robot_pos))
            lidar_start = time.perf_counter()
            State, max_ray_angle = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size, end_point)

            if step == 0:
                angle = max_ray_angle
                rays_this_rollout += network.last_rays_cast
                sensing_units += network.last_rays_cast * ray_range / 10.0
                State, _ = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size, end_point)
               
            lidar_seconds_this_rollout += time.perf_counter() - lidar_start
            
            rays_this_rollout += network.last_rays_cast
            sensing_units += network.last_rays_cast * ray_range / 10.0    # rays x range, relative to a 10 px reference
            
            # Get action
            inference_start = time.perf_counter()
            with torch.no_grad():
                mean, _, value = network.forward(State)
                Action = network._squash_action(mean).cpu().numpy()
            inference_seconds += time.perf_counter() - inference_start

            # params (old angle, angle, etc.)
            old_angle = float(angle)
            new_angle, step_size = old_angle + float(Action[0]), float(Action[1])

            turn_limit = VARtargetMaxTurn
            turn = (new_angle - old_angle + math.pi) % (2 * math.pi) - math.pi
            turn_excess = max(0.0, abs(turn) - turn_limit)    # how far past the limit the policy asked to turn
            turn = max(-turn_limit, min(turn_limit, turn))
            new_angle = (old_angle + turn + math.pi) % (2 * math.pi) - math.pi
            network.last_action = [turn / math.pi, step_size / network.max_step]

            # Virtually update robot stats 
            new_pos = (robot_pos + step_size * np.array([np.cos(new_angle), np.sin(new_angle)])).astype(np.float32)
            visit_pixel = tuple(int(np.clip(np.floor(new_pos[i]), 0, PixelVisits.shape[i] - 1)) for i in range(2))
            current_pixel_visits = PixelVisits[visit_pixel]         # times this pixel was entered before now
            PixelVisits[visit_pixel] += 1
            revisit_steps += int(current_pixel_visits > 0)
            robot_path.append(new_pos.copy())
            path_length += float(np.linalg.norm(new_pos - robot_pos))
            physics_energy += move_joules(turn, step_size)
            angle_diff_step = abs((new_angle - old_angle + math.pi) % (2 * math.pi) - math.pi)
            total_turn_delta += angle_diff_step # turn sharpness

            # Get state info
            run_into_wall = bool(RunIntoWall(robot_pos, walls, new_pos[0] - robot_pos[0], new_pos[1] - robot_pos[1]))
            in_dead_end, region_pos = network.InDeadEnd(grid_distance, new_pos, walls_np)
            region_tuple = tuple(region_pos.tolist())

            if region_tuple == start_region or region_tuple == goal_region:      
                in_dead_end = False
            region_unexplored = region_tuple not in RegionsExplored
            if region_unexplored:
                RegionsExplored.add(region_tuple)

            next_distance_finish = float(np.linalg.norm(end_point - new_pos))
            distance_progress = distance_finish - next_distance_finish
            in_goal_pixel = bool(np.all(np.floor(new_pos / grid_distance) == np.floor(end_point / grid_distance)))
            goal_distance = 0.0 if in_goal_pixel else next_distance_finish

            min_ray_px = min(State[:int(VARNumberOfRayCasts)]) ** 2 * RAY_SCALE

            # Get reward
            Reward, done, maze_solved = network.reward(
                distance_progress, run_into_wall, in_dead_end, goal_distance,
                region_unexplored, VARRayDistance, new_pos, step_size, energy_ramp_rate,
                old_angle, new_angle, maze_size, done, min_ray_px, turn_excess, current_pixel_visits
            )

            # prevent stalling
            if not done and len(robot_path) > STALL_WINDOW and \
                    np.linalg.norm(new_pos - robot_path[-1 - STALL_WINDOW]) < STALL_MIN_DISPLACEMENT:
                Reward -= STALL_END_PENALTY
                done, stalled = True, True

            episode_reward += Reward
            episode_length += 1
            penalty_sum += network.last_total_penalty

            robot_pos, angle = new_pos, new_angle
            if done:
                termination = "success" if maze_solved else ("collision" if run_into_wall else ("stalled" if stalled else ("dead_end" if in_dead_end else "battery")))
                break

        past += int(maze_solved)
        not_past += int(not maze_solved)
        success_rate = past / (past + not_past) 

        # accuracy using A*
        ACCURACY = network.GetRobotAccuracy(np.asarray(A_star_path), np.asarray(robot_path))
        path_efficiency = min(1.0, optimal_length / max(path_length, 1e-6)) if maze_solved else 0.0   # success-weighted shortest/actual path
        energy_per_length = network.EnergyUsed / max(path_length, 1e-6)
        Episode_Payload = [episode_count, episode_reward, episode_length, int(maze_solved), success_rate,
                           network.EnergyUsed, len(RegionsExplored), "completed" if done else "partial",
                           penalty_sum, path_length, total_turn_delta, lidar_seconds_this_rollout,
                           inference_seconds, rays_this_rollout, termination,
                           maze_size, episode_x, episode_stage, optimal_length, path_efficiency, ACCURACY, energy_per_length,
                           sensing_units, network.max_step, revisit_steps, time.perf_counter() - start_time, physics_energy]

        # Log data
        network.LogDataToCSV(Episode_Payload, None, None, "ep", agnt, directory, test)
        #network.LogDataToCSV(pixel_grid, ACCURACY, None, "mze", agnt, directory, test)
        episode_count += 1

    return network