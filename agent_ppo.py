import csv
import json
import math
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
from pathlib import Path

from actor_critic import (
    ActorCritic,
    ADAPT_TURN_WEIGHT,
    DEAD_END_IS_TERMINAL,
    ENERGY_REWARD_WEIGHT,
    TURN_REWARD_WEIGHT,
    TIME_REWARD_WEIGHT,
)
from maze_utils import raycaster_env, _fov_to_radians, _build_bsp_tree

# Flags
ACTION_DELAY_SECONDS = 0.0
ROLLOUTS = 750
SEED = 42
MIN_ROLLOUT_STEPS = 75000 # must be positive
PPO_EPOCHS = 16
MINIBATCH_SIZE = 64
TARGET_KL = 0.02
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
CURRICULUM_ENABLED = True
REWARD_DEBUG_EVERY = 20

def _compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95): # advantages goes to loss
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

    # training loop, moved to a function
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

            # batched AI. This is what we are studying
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
                network.log_std.clamp_(-3.0, 1.0)
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

# main
def StartAgent(test, NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType, VARtargetMaxTurn, agnt, VARFOV, VARRayDistance, *, seed=SEED, rollouts=None, output_dir=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rollouts = ROLLOUTS if rollouts is None else int(rollouts)

    directory = Path(output_dir) if output_dir is not None else Path("runs") / f"agent_{agnt}_seed_{seed}_{time.time_ns()}"
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / f"RunConfig_{agnt}.json"
    x = 51
    success_rate = 0.0
    episode_count = 0
    past = 0
    not_past = 0

    # classes
    network = ActorCritic(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, VARFOV, NNinputs)
    network.InitializeWeights(VARWeightInitType)

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
                  ppo_epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE, target_kl=TARGET_KL,
                  learning_rate=LEARNING_RATE, entropy_coef=ENTROPY_COEF,
                  dead_end_terminal=DEAD_END_IS_TERMINAL, torch_version=torch.__version__,
                  numpy_version=np.__version__, python_version=sys.version)
    
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    headers = {
        "EpisodeData": ["episode", "reward", "steps", "success", "success_rate", "raw_energy", "regions", "status",
                        "penalty_sum", "path_length", "turn_radians", "lidar_seconds", "inference_seconds", "rays_cast", "termination"],
        "ActionData": ["angle", "step", "value", "reward", "total_penalty", "raw_energy", "energy_penalty", "turn_penalty", "time_penalty"],
        "TrainingData": ["episode_count", "success_rate", "total_loss", "actor_loss", "critic_huber_loss", "bounded_entropy_mc", "advantage_mean", "kl_before_step", "clip_fraction", "grad_norm_before_clip"],
        "UpdateData": ["episode_count", "transitions", "minibatch_updates", "final_kl", "kl_stopped", "explained_variance", "training_seconds"],
    }

    for name, header in headers.items():
        with open(directory / f"{name}_{test}{agnt}.csv", "x", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    capacity = MIN_ROLLOUT_STEPS + network.MaxSteps
    network.clear_memory(capacity)
    buffered_steps = 0
    buffered_advantages, buffered_returns = [], []
    buffer_turn_total = 0.0
    for _rollout in range(rollouts):

        # get training example
        start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd)
        robot_pos = np.array(start_point, dtype=np.float32)
        end_point = np.array(end_point, dtype=np.float32)
        walls_np, BSP = _build_bsp_tree(walls)
        robot_pos = robot_pos + 0.5
        end_point = end_point + 0.5

        # Initializations
        angle = float((7 * math.pi) / 4)
        A_star_path = Normal_A_star_search(np.array(pixel_grid), list(start_point), list(end_point))
        total_turn_delta = 0.0
        episode_start = buffered_steps
        RegionsExplored = set()

        start_region = None
        if grid_distance > 0:
            start_region = tuple((np.floor(robot_pos / grid_distance) * grid_distance).tolist())
            goal_region = tuple((np.floor(end_point / grid_distance) * grid_distance).tolist())
            RegionsExplored.add(start_region)

        network.EnergyUsed = 0.0
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
            State = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size, end_point)
            lidar_seconds_this_rollout += network.time_cost
            rays_this_rollout += network.last_rays_cast
            distance_finish = float(np.linalg.norm(end_point - robot_pos))

            # Get action
            inference_start = time.perf_counter()
            Action, log_prob, value, LatentAction, _ = network.get_action_and_value(State)
            inference_seconds += time.perf_counter() - inference_start

            # Update the agent viewer
            #Dashboard.send([[Action]], walls)

            # params (old angle, angle, etc.)
            old_angle = float(angle)
            new_angle, step_size = float(Action[0]), float(Action[1])

            # Virtually update robot stats without directional floor bias.
            new_pos = (robot_pos + step_size * np.array([np.cos(new_angle), np.sin(new_angle)])).astype(np.float32)
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

            # Get reward
            next_distance_finish = float(np.linalg.norm(end_point - new_pos))
            distance_progress = distance_finish - next_distance_finish
            in_goal_pixel = bool(np.all(np.floor(new_pos / grid_distance) == np.floor(end_point / grid_distance)))
            goal_distance = 0.0 if in_goal_pixel else next_distance_finish

            Reward, done, maze_solved = network.reward(
                distance_progress, run_into_wall, in_dead_end, goal_distance,
                region_unexplored, robot_pos, new_pos, step_size, success_rate,
                old_angle, new_angle, maze_size, done
            )

            episode_reward += Reward
            episode_length += 1
            penalty_sum += network.last_total_penalty

            # store episode data
            network.store_transition(State, LatentAction, Reward, log_prob, done, value, buffered_steps)
            buffered_steps += 1

            # action data log
            network.LogDataToCSV(None, [float(Action[0]), float(Action[1]), float(value), Reward,
                                       network.last_total_penalty, network.last_raw_energy,
                                       network.last_energy_component, network.last_turn_component,
                                       network.last_time_component], None, "act", agnt, directory, test)
            # update the robot (angle / position)
            robot_pos, angle = new_pos, new_angle
            if done:
                termination = "success" if maze_solved else ("collision" if run_into_wall else "dead_end")
                break
            if ACTION_DELAY_SECONDS > 0:
                time.sleep(ACTION_DELAY_SECONDS)

        past += int(maze_solved)
        not_past += int(not maze_solved)
        if maze_solved and CURRICULUM_ENABLED: # added a ciriculum enabler to track early stability
            x += 1

        success_rate = past / (past + not_past) # sucessrate (currently 0 in early training)

        # accuracy using A*
        ACCURACY = network.GetRobotAccuracy(np.asarray(A_star_path), np.asarray(robot_path))

        Episode_Payload = [episode_count, episode_reward, episode_length, int(maze_solved), success_rate,
                           network.EnergyUsed, len(RegionsExplored), "completed" if done else "partial",
                           penalty_sum, path_length, total_turn_delta, lidar_seconds_this_rollout,
                           inference_seconds, rays_this_rollout, termination]

        # Log data
        network.LogDataToCSV(Episode_Payload, None, None, "ep", agnt, directory, test)
        network.LogDataToCSV(pixel_grid, ACCURACY, None, "mze", agnt, directory, test)
        episode_count += 1

        # Comput GAE and PPO / other opitimizations
        if done:
            next_value = torch.tensor(0.0, dtype=torch.float32)
        else:
            with torch.no_grad():
                FinalState = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size, end_point)
                _, _, next_value = network.forward(FinalState)

        episode_slice = slice(episode_start, buffered_steps)
        advantages, returns = _compute_gae(network.rewards[episode_slice], network.values[episode_slice],
                                          network.dones[episode_slice], next_value)
        buffered_advantages.append(advantages)
        buffered_returns.append(returns)
        buffer_turn_total += total_turn_delta
        if buffered_steps < MIN_ROLLOUT_STEPS and _rollout != rollouts - 1:
            continue

        # TRAINING PPO LOOP
        update_start = time.perf_counter() # (todo measure time and for biggest bottlenecks)
        diagnostics, final_kl, kl_stopped, explained_variance = _ppo_update(
            network, optimizer, ppo_parameters, network.states[:buffered_steps], network.actions[:buffered_steps],
            network.log_probs[:buffered_steps], torch.cat(buffered_returns), torch.cat(buffered_advantages))

        training_seconds = time.perf_counter() - update_start
        for row in diagnostics:
            network.LogDataToCSV([episode_count, success_rate] + row, None, None, "trg", agnt, directory, test)

        with open(directory / f"UpdateData_{agnt}.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([episode_count, buffered_steps, len(diagnostics), final_kl, int(kl_stopped), explained_variance, training_seconds])

        # Update weights adam
        if w_optimizer is not None:
            avg_turn = buffer_turn_total / buffered_steps
            w_turn_loss = -network.w_turn * (avg_turn - network.target_max_turn)
            w_optimizer.zero_grad()
            w_turn_loss.backward()
            w_optimizer.step()
            with torch.no_grad():
                network.log_w_turn.clamp_(math.log(1e-4), math.log(10.0))

        if REWARD_DEBUG_EVERY and (episode_count % REWARD_DEBUG_EVERY == 0 or _rollout == rollouts - 1):
            print(f"episode={episode_count} success={success_rate:.3f} steps={buffered_steps} KL={final_kl:.4f}")

        network.clear_memory(capacity) # return to next episode
        buffered_steps = 0
        buffered_advantages, buffered_returns = [], []
        buffer_turn_total = 0.0

    # added a way to save the model's state
    torch.save({"model_state_dict": network.state_dict(), "config": config}, directory / f"Policy_{agnt}.pt")
    network.output_directory = str(directory)
    return network

