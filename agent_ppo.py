import csv
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
from training_example_generator import get_new_training_example
from A_star_search import Normal_A_star_search
from RunIntoWall import RunIntoWall
from agent_viewer import Dashboard
import sys
import random
from collections import deque
from pathlib import Path

from actor_critic import (
    ActorCritic,
    ADAPT_TURN_WEIGHT,
    DEAD_END_IS_TERMINAL,
    ENERGY_REWARD_WEIGHT,
    TURN_REWARD_WEIGHT,
    TIME_REWARD_WEIGHT,
    MAX_STEP_START,
    MAX_STEP_END,
    STEP_RAMP_SUCCESS,
    REWARD_CLIP,
    STEP_TIME_COST,
    CLEARANCE_WEIGHT,
    CLEARANCE_DISTANCE,
    VISIT_PENALTY,
    VISIT_CAP,
    EXPLORE_BONUS,
    STALL_STEP_SIZE,
    STALL_STEP_PENALTY,
    OVER_BUDGET_PENALTY,
    RAY_SCALE,
    LOG_STD_MIN,
    HARD_ENERGY_LIMIT,
)
from maze_utils import raycaster_env, _fov_to_radians, _build_bsp_tree

# Flags
ACTION_DELAY_SECONDS = 0.0
ROLLOUTS = 3000
SEED = 42
MIN_ROLLOUT_STEPS = 256 # must be positive
PPO_EPOCHS = 8
MINIBATCH_SIZE = 64
TARGET_KL = 0.02
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
REWARD_SCALE = 0.2
CURRICULUM_ENABLED = True
TURN_LIMIT_SUCCESS = 0.4  
MIN_ROLLOUT_STEPS_LEARNING = 512   
CURRICULUM_WINDOW = 50             
CURRICULUM_PROMOTE = 0.6           
CURRICULUM_DEMOTE = 0.1            
CURRICULUM_STAGES = [30, 40, 50, 51, 52, 54, 55, 56, 53]
KEEP_RAMP_FROM_STAGE = 2
STALL_WINDOW = 20                  
STALL_MIN_DISPLACEMENT = 2.0     
STALL_END_PENALTY = 1.0
SUMMARY_SUCCESS_WINDOW = 100    

def _compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95): 
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=values.dtype)
    for t in reversed(range(len(rewards))):
        next_value_t = next_value if t == len(rewards) - 1 else values[t + 1]
        done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value_t * done - values[t]
        next_advantage = delta + gamma * lam * done * next_advantage
        advantages[t] = next_advantage

    return advantages, advantages + values

def _ppo_update(network, optimizer, ppo_parameters, states, actions, log_probs, returns, advantages):
    advantages = advantages.detach().clone()
    returns, log_probs = returns.detach(), log_probs.detach()
    if len(advantages) > 1:
        std = advantages.std(unbiased=False)
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / std

    with torch.no_grad():
        old_mean, old_std, old_values = network.forward(states)
        old_dist = dist.Normal(old_mean.clone(), old_std.clone())
    diagnostics = []
    stopped = False

    # PPO Training loop
    for epoch in range(PPO_EPOCHS):
        shuffled_indices = torch.randperm(len(states))
        for i in range(0, len(states), MINIBATCH_SIZE):
            with torch.no_grad():
                mean_now, std_now, _ = network.forward(states)
                kl = dist.kl_divergence(old_dist, dist.Normal(mean_now, std_now)).sum(-1).mean() # Kl for policy change

            if not torch.isfinite(kl):
                raise FloatingPointError("Non-finite policy KL.")
            if kl > TARGET_KL:
                stopped = True
                break

            # batched allows for better handling
            batch = shuffled_indices[i:i + MINIBATCH_SIZE]
            new_log_prob, entropy, batch_value = network.evaluate_actions(states[batch], actions[batch])
            probability_ratio = torch.exp(new_log_prob - log_probs[batch])
            surrogate_1 = probability_ratio * advantages[batch]
            surrogate_2 = torch.clamp(probability_ratio, 0.8, 1.2) * advantages[batch]

            # Loss
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
            critic_loss = F.smooth_l1_loss(batch_value, returns[batch])
            entropy_mean = entropy.mean()
            total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy_mean

            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite PPO loss.")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ppo_parameters, 0.5, error_if_nonfinite=True)
            optimizer.step()

            with torch.no_grad():
                network.log_std.clamp_(LOG_STD_MIN, 1.0)
                clip_fraction = ((probability_ratio - 1.0).abs() > 0.2).float().mean()

            diagnostics.append([
                total_loss.item(), actor_loss.item(), critic_loss.item(), entropy_mean.item(),
                advantages[batch].mean().item(), kl.item(), clip_fraction.item(), float(grad_norm),
            ])
        if stopped:
            break

    with torch.no_grad():
        new_mean, new_std, new_values = network.forward(states)
        final_kl = dist.kl_divergence(old_dist, dist.Normal(new_mean, new_std)).sum(-1).mean().item() # KL controlls how much the policy can change
        variance = returns.var(unbiased=False)
        explained_variance = (1.0 - (returns - new_values).var(unbiased=False) / variance).item() if variance > 1e-8 else None

    return diagnostics, final_kl, stopped or final_kl > TARGET_KL, explained_variance

def _write_summary(path, history, config, start_time):
    def mean(values):
        values = [v for v in values if v is not None and math.isfinite(v)]
        return float(np.mean(values)) if values else None

    wins = [h for h in history if h["success"]]
    last = history[-SUMMARY_SUCCESS_WINDOW:]
    last_wins = [h for h in last if h["success"]]
    win_energy = sorted(h["energy"] for h in wins)
    summary = dict(
        episodes=len(history), wall_seconds=time.perf_counter() - start_time,
        last_window_success_rate=mean([h["success"] for h in last]),
        last_window_path_efficiency=mean([h["spl"] for h in last]),
        last_window_accuracy=mean([h["accuracy"] for h in last_wins]),
        last_window_energy_per_success=mean([h["energy"] for h in last_wins]),
        last_window_energy_per_length=mean([h["energy_per_length"] for h in last_wins]),
        last_window_rays_per_episode=mean([h["rays"] for h in last]),
        last_window_steps_per_episode=mean([h["steps"] for h in last]),
        total_energy=sum(h["energy"] for h in history), total_rays=sum(h["rays"] for h in history),
        total_sensing_units=sum(h["sensing"] for h in history),
        total_lidar_seconds=sum(h["lidar_seconds"] for h in history),
        total_inference_seconds=sum(h["inference_seconds"] for h in history),
        successes=len(wins),
        success_energy_median=float(np.median(win_energy)) if win_energy else None,
        success_energy_p90=float(np.percentile(win_energy, 90)) if win_energy else None,
        suggested_energy_budget=round(2.0 * float(np.median(win_energy)), 1) if len(win_energy) >= 5 else None,
        param_count=config["param_count"], actor_param_count=config["actor_param_count"],
        actor_macs_per_inference=config["actor_macs_per_inference"],
    )
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

# checkpoint / resume (main.py sets STOP_AT and STOP_FILE for each agent)
CHECKPOINT_EVERY = 1000     # episodes between automatic checkpoints
STOP_AT = None              # unix time after which the agent saves a checkpoint and stops
STOP_FILE = None            # if this file exists the agent saves a checkpoint and stops

def _trim_csv(path, column, keep):
    # drop log rows written after the checkpoint (only matters after a crash)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows = rows[:1] + [r for r in rows[1:] if keep(float(r[column]))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

# main
def StartAgent(test, NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType, VARtargetMaxTurn, agnt, VARFOV, VARRayDistance, *, seed=SEED, rollouts=None, output_dir=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rollouts = ROLLOUTS if rollouts is None else int(rollouts)

    directory = Path(output_dir) if output_dir is not None else Path("runs") / f"agent_{agnt}_seed_{seed}_{time.time_ns()}"
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / f"RunConfig_{agnt}.json"
    x = CURRICULUM_STAGES[0]
    success_rate = 0.0
    episode_count = 0
    past = 0
    not_past = 0
    stage = 0                                 
    recent_outcomes = deque(maxlen=CURRICULUM_WINDOW)   
    stage_outcomes = deque(maxlen=CURRICULUM_WINDOW)   
    ramp_rate = 0.0     
    energy_ramp_rate = 0.0
    history = []        
    start_time = time.perf_counter()

    # classes
    network = ActorCritic(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, VARFOV, NNinputs)
    network.InitializeWeights(VARWeightInitType)
    network.set_max_step(ramp_rate)

    # config
    ppo_parameters = [p for name, p in network.named_parameters() if p.requires_grad and name != "log_w_turn"]
    optimizer = torch.optim.Adam(ppo_parameters, lr=LEARNING_RATE, eps=1e-5)
    w_optimizer = torch.optim.Adam([network.log_w_turn], lr=1e-3) if ADAPT_TURN_WEIGHT else None

    # added config for params
    config = dict(seed=seed, rollouts=rollouts, h1=NNh1, h2=NNh2, inputs=NNinputs,
                  rays=VARNumberOfRayCasts, fov_radians=network.FOV, ray_distance=VARRayDistance,
                  energy_budget=VARAllowedEnergy, energy_scale=VARAllowedEnergy / network.MaxSteps,
                  energy_weight=ENERGY_REWARD_WEIGHT, turn_weight=TURN_REWARD_WEIGHT,
                  time_weight=TIME_REWARD_WEIGHT, adaptive_turn=ADAPT_TURN_WEIGHT,
                  target_max_turn=VARtargetMaxTurn, curriculum=CURRICULUM_ENABLED,
                  initial_difficulty=x, k=network.k.item(), sd=network.sd.item(),
                  ray_offset=network.RayOffset.item(), weight_init=VARWeightInitType,
                  max_steps=network.MaxSteps, min_rollout_steps=MIN_ROLLOUT_STEPS,
                  max_step_start=MAX_STEP_START, max_step_end=MAX_STEP_END, step_ramp_success=STEP_RAMP_SUCCESS,
                  min_rollout_steps_learning=MIN_ROLLOUT_STEPS_LEARNING, curriculum_window=CURRICULUM_WINDOW,
                  curriculum_promote=CURRICULUM_PROMOTE, curriculum_demote=CURRICULUM_DEMOTE,
                  curriculum_stages=CURRICULUM_STAGES, stall_window=STALL_WINDOW,
                  stall_min_displacement=STALL_MIN_DISPLACEMENT, stall_end_penalty=STALL_END_PENALTY,
                  reward_clip=REWARD_CLIP, step_time_cost=STEP_TIME_COST, clearance_weight=CLEARANCE_WEIGHT,
                  clearance_distance=CLEARANCE_DISTANCE, visit_penalty=VISIT_PENALTY, visit_cap=VISIT_CAP,
                  explore_bonus=EXPLORE_BONUS, stall_step_size=STALL_STEP_SIZE, stall_step_penalty=STALL_STEP_PENALTY,
                  over_budget_penalty=OVER_BUDGET_PENALTY,
                  reward_scale=REWARD_SCALE, hard_energy_limit=HARD_ENERGY_LIMIT, log_std_min=LOG_STD_MIN, ray_scale=RAY_SCALE,
                  param_count=sum(p.numel() for p in network.parameters()),
                  actor_param_count=sum(p.numel() for m in (network.act1, network.act2, network.out) for p in m.parameters()),
                  actor_macs_per_inference=int(network.inputs * NNh1 + NNh1 * NNh2 + NNh2 * 2),
                  ppo_epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE, target_kl=TARGET_KL,
                  learning_rate=LEARNING_RATE, entropy_coef=ENTROPY_COEF,
                  dead_end_terminal=DEAD_END_IS_TERMINAL, torch_version=torch.__version__,
                  numpy_version=np.__version__, python_version=sys.version)
    
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    headers = {
        "EpisodeData": ["episode", "reward", "steps", "success", "success_rate", "raw_energy", "regions", "status",
                        "penalty_sum", "path_length", "turn_radians", "lidar_seconds", "inference_seconds", "rays_cast", "termination",
                        "maze_size", "difficulty_x", "stage", "optimal_length", "path_efficiency", "accuracy", "energy_per_length",
                        "sensing_units", "max_step_cap", "revisit_steps", "elapsed_seconds", "rolling_success"],
        "ActionData": ["angle", "step", "value", "reward", "total_penalty", "raw_energy", "energy_penalty", "turn_penalty", "time_penalty"],
        "TrainingData": ["episode_count", "success_rate", "total_loss", "actor_loss", "critic_huber_loss", "bounded_entropy_mc", "advantage_mean", "kl_before_step", "clip_fraction", "grad_norm_before_clip"],
        "UpdateData": ["episode_count", "transitions", "minibatch_updates", "final_kl", "kl_stopped", "explained_variance", "training_seconds",
                       "episodes_in_buffer", "mean_episode_reward", "mean_episode_steps", "policy_entropy", "log_std_angle", "log_std_step",
                       "rolling_success", "ramp_rate", "max_step_cap", "stage", "rollout_target"],
    }

    # resume from a checkpoint if there is one in this folder
    checkpoint_path = directory / "checkpoint.pt"
    resumed = checkpoint_path.exists()
    if resumed:
        ck = torch.load(checkpoint_path, weights_only=False)
        network.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if w_optimizer is not None:
            w_optimizer.load_state_dict(ck["w_optimizer"])
        x, stage, ramp_rate, energy_ramp_rate = ck["x"], ck["stage"], ck["ramp_rate"], ck["energy_ramp_rate"]
        past, not_past, success_rate, episode_count = ck["past"], ck["not_past"], ck["success_rate"], ck["episode_count"]
        recent_outcomes.extend(ck["recent_outcomes"])
        stage_outcomes.extend(ck["stage_outcomes"])
        history = ck["history"]
        start_time = time.perf_counter() - ck["elapsed"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        network.set_max_step(ramp_rate)
        _trim_csv(directory / f"EpisodeData_{test}{agnt}.csv", 0, lambda v: v < episode_count)
        _trim_csv(directory / f"TrainingData_{test}{agnt}.csv", 0, lambda v: v <= episode_count)
        _trim_csv(directory / f"UpdateData_{test}{agnt}.csv", 0, lambda v: v <= episode_count)
    last_checkpoint = episode_count

    if not resumed:
        for name, header in headers.items():
            with open(directory / f"{name}_{test}{agnt}.csv", "x", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    capacity = max(MIN_ROLLOUT_STEPS, MIN_ROLLOUT_STEPS_LEARNING) + network.MaxSteps
    network.clear_memory(capacity)
    buffered_steps = 0
    buffered_advantages, buffered_returns = [], []
    buffer_turn_total = 0.0
    buffer_episode_rewards, buffer_episode_steps = [], []
    for _rollout in range(episode_count, rollouts):
        # get training example
        start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd)
        robot_pos = np.array(start_point, dtype=np.float32)
        end_point = np.array(end_point, dtype=np.float32)
        walls_np, BSP = _build_bsp_tree(walls)
        robot_pos = robot_pos + 0.5
        end_point = end_point + 0.5
        episode_x, episode_stage = x, stage

        # Initializations
        angle = 0.0
        A_star_path = Normal_A_star_search(np.array(pixel_grid), [int(v) for v in start_point], [int(v) for v in np.floor(end_point)])
        A_star_points = np.asarray(A_star_path, dtype=float)
        optimal_length = float(np.linalg.norm(np.diff(A_star_points, axis=0), axis=1).sum()) if len(A_star_points) > 1 else 0.0
        total_turn_delta = 0.0
        episode_start = buffered_steps
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
            Action, log_prob, value, LatentAction, _ = network.get_action_and_value(State)
            inference_seconds += time.perf_counter() - inference_start

            # Update the agent viewer
            #Dashboard.send([[Action]], walls)

            # params (old angle, angle, etc.)
            old_angle = float(angle)
            new_angle, step_size = old_angle + float(Action[0]), float(Action[1])

            # added a way for early training for energy efficiency to 
            turn_limit = math.pi - (math.pi - VARtargetMaxTurn) * min(1.0, ramp_rate / TURN_LIMIT_SUCCESS)
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

            # Get reward
            next_distance_finish = float(np.linalg.norm(end_point - new_pos))
            distance_progress = distance_finish - next_distance_finish
            in_goal_pixel = bool(np.all(np.floor(new_pos / grid_distance) == np.floor(end_point / grid_distance)))
            goal_distance = 0.0 if in_goal_pixel else next_distance_finish

            # nearest wall the LiDAR saw this step, in pixels (rays outside the FOV hold the max distance, so they never win)
            min_ray_px = min(State[:int(VARNumberOfRayCasts)]) ** 2 * RAY_SCALE

            Reward, done, maze_solved = network.reward(
                distance_progress, run_into_wall, in_dead_end, goal_distance,
                region_unexplored, VARRayDistance, new_pos, step_size, energy_ramp_rate,
                old_angle, new_angle, maze_size, done, min_ray_px, turn_excess, current_pixel_visits
            )

            # end hopeless episodes early: almost no net movement over the last STALL_WINDOW steps
            if not done and len(robot_path) > STALL_WINDOW and \
                    np.linalg.norm(new_pos - robot_path[-1 - STALL_WINDOW]) < STALL_MIN_DISPLACEMENT:
                Reward -= STALL_END_PENALTY
                done, stalled = True, True

            episode_reward += Reward
            episode_length += 1
            penalty_sum += network.last_total_penalty

            # store episode data
            network.store_transition(State, LatentAction, Reward * REWARD_SCALE, log_prob, done, value, buffered_steps)
            buffered_steps += 1

            # action data log
            #network.LogDataToCSV(None, [float(Action[0]), float(Action[1]), float(value), Reward,
           #                            network.last_total_penalty, network.last_raw_energy,
            #                           network.last_energy_component, network.last_turn_component,
            #                           network.last_time_component], None, "act", agnt, directory, test)
            # update the robot (angle / position)
            robot_pos, angle = new_pos, new_angle
            if done:
                termination = "success" if maze_solved else ("collision" if run_into_wall else ("stalled" if stalled else ("dead_end" if in_dead_end else "battery")))
                break
            if ACTION_DELAY_SECONDS > 0:
                time.sleep(ACTION_DELAY_SECONDS)

        past += int(maze_solved)
        not_past += int(not maze_solved)
        success_rate = past / (past + not_past) 

        # Cirriculum
        recent_outcomes.append(int(maze_solved))
        rolling_success = sum(recent_outcomes) / len(recent_outcomes)

        if len(recent_outcomes) == CURRICULUM_WINDOW:
            ramp_rate = max(ramp_rate, rolling_success)
            energy_ramp_rate = max(energy_ramp_rate, rolling_success)
        if CURRICULUM_ENABLED:   
            stage_outcomes.append(int(maze_solved))
            if len(stage_outcomes) == CURRICULUM_WINDOW:
                stage_rate = sum(stage_outcomes) / CURRICULUM_WINDOW
                if stage_rate >= CURRICULUM_PROMOTE and stage < len(CURRICULUM_STAGES) - 1:
                    stage += 1
                    stage_outcomes.clear()
                    if stage - 1 < KEEP_RAMP_FROM_STAGE:
                        ramp_rate = 0.0
                    recent_outcomes.clear()
                elif stage_rate <= CURRICULUM_DEMOTE and stage > 0:
                    stage -= 1
                    stage_outcomes.clear()
                    if stage < KEEP_RAMP_FROM_STAGE:
                        ramp_rate = 0.0
                    recent_outcomes.clear()
                x = CURRICULUM_STAGES[stage]

        # accuracy using A*
        ACCURACY = network.GetRobotAccuracy(np.asarray(A_star_path), np.asarray(robot_path))
        path_efficiency = min(1.0, optimal_length / max(path_length, 1e-6)) if maze_solved else 0.0   # success-weighted shortest/actual path
        energy_per_length = network.EnergyUsed / max(path_length, 1e-6)
        history.append(dict(success=int(maze_solved), spl=path_efficiency, energy=float(network.EnergyUsed), accuracy=float(ACCURACY),
                            energy_per_length=energy_per_length, rays=rays_this_rollout, steps=episode_length, sensing=sensing_units,
                            lidar_seconds=lidar_seconds_this_rollout, inference_seconds=inference_seconds))
        buffer_episode_rewards.append(episode_reward)
        buffer_episode_steps.append(episode_length)
        Episode_Payload = [episode_count, episode_reward, episode_length, int(maze_solved), success_rate,
                           network.EnergyUsed, len(RegionsExplored), "completed" if done else "partial",
                           penalty_sum, path_length, total_turn_delta, lidar_seconds_this_rollout,
                           inference_seconds, rays_this_rollout, termination,
                           maze_size, episode_x, episode_stage, optimal_length, path_efficiency, ACCURACY, energy_per_length,
                           sensing_units, network.max_step, revisit_steps, time.perf_counter() - start_time, rolling_success]

        # Log data
        network.LogDataToCSV(Episode_Payload, None, None, "ep", agnt, directory, test)
        #network.LogDataToCSV(pixel_grid, ACCURACY, None, "mze", agnt, directory, test)
        episode_count += 1

        # Comput GAE and PPO / other opitimizations
        if done:
            next_value = torch.tensor(0.0, dtype=torch.float32)
        else:
            with torch.no_grad():
                FinalState, _ = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size, end_point)
                _, _, next_value = network.forward(FinalState)

        episode_slice = slice(episode_start, buffered_steps)
        advantages, returns = _compute_gae(network.rewards[episode_slice], network.values[episode_slice],
                                          network.dones[episode_slice], next_value)
        buffered_advantages.append(advantages)
        buffered_returns.append(returns)
        buffer_turn_total += total_turn_delta
        rollout_target = MIN_ROLLOUT_STEPS_LEARNING if sum(recent_outcomes) > 0 else MIN_ROLLOUT_STEPS
        if buffered_steps < rollout_target and _rollout != rollouts - 1:
            continue

        # TRAINING PPO LOOP
        update_start = time.perf_counter() # (todo measure time and for biggest bottlenecks)
        diagnostics, final_kl, kl_stopped, explained_variance = _ppo_update(
            network, optimizer, ppo_parameters, network.states[:buffered_steps], network.actions[:buffered_steps],
            network.log_probs[:buffered_steps], torch.cat(buffered_returns), torch.cat(buffered_advantages))

        training_seconds = time.perf_counter() - update_start
        for row in diagnostics:
            network.LogDataToCSV([episode_count, success_rate] + row, None, None, "trg", agnt, directory, test)

        with open(directory / f"UpdateData_{test}{agnt}.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([episode_count, buffered_steps, len(diagnostics), final_kl, int(kl_stopped), explained_variance, training_seconds,
                                    len(buffer_episode_rewards), float(np.mean(buffer_episode_rewards)), float(np.mean(buffer_episode_steps)),
                                    float(np.mean([row[3] for row in diagnostics])) if diagnostics else None,
                                    float(network.log_std[0]), float(network.log_std[1]), rolling_success, ramp_rate,
                                    network.max_step, stage, rollout_target])
        # Update weights adam
        if w_optimizer is not None:
            avg_turn = buffer_turn_total / buffered_steps
            w_turn_loss = -network.w_turn * (avg_turn - network.target_max_turn)
            w_optimizer.zero_grad()
            w_turn_loss.backward()
            w_optimizer.step()
            with torch.no_grad():
                network.log_w_turn.clamp_(math.log(1e-4), math.log(10.0))
        network.clear_memory(capacity) 
        network.set_max_step(ramp_rate) 
        buffered_steps = 0
        buffered_advantages, buffered_returns = [], []
        buffer_turn_total = 0.0
        buffer_episode_rewards, buffer_episode_steps = [], []
        _write_summary(directory / f"Summary_{test}{agnt}.json", history, config, start_time)

        # checkpoint: every CHECKPOINT_EVERY episodes, and when it is time to stop
        stopping = (STOP_AT is not None and time.time() >= STOP_AT) or (STOP_FILE is not None and os.path.exists(STOP_FILE))
        if _rollout != rollouts - 1 and (stopping or episode_count - last_checkpoint >= CHECKPOINT_EVERY):
            torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict(),
                        "w_optimizer": w_optimizer.state_dict() if w_optimizer is not None else None,
                        "x": x, "stage": stage, "ramp_rate": ramp_rate, "energy_ramp_rate": energy_ramp_rate,
                        "past": past, "not_past": not_past, "success_rate": success_rate, "episode_count": episode_count,
                        "recent_outcomes": list(recent_outcomes), "stage_outcomes": list(stage_outcomes), "history": history,
                        "elapsed": time.perf_counter() - start_time, "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, directory / "checkpoint.tmp")
            os.replace(directory / "checkpoint.tmp", checkpoint_path)
            last_checkpoint = episode_count
            if stopping:
                return

    # added a way to save the model's state
    torch.save({"model_state_dict": network.state_dict(), "config": config, "max_step": network.max_step}, directory / f"Policy_{agnt}.pt")
    network.output_directory = str(directory)
    _write_summary(directory / f"Summary_{test}{agnt}.json", history, config, start_time)
    return network