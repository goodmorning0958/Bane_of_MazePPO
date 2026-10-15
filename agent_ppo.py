import csv
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from training_example_generator import get_new_training_example
from SImulated_BSP_LiDAR import Tree
from A_star_search import Normal_A_star_search
from RunIntoWall import RunIntoWall
from PPO_action_converter import ConvertRobotAction
from agent_viewer import Dashboard
import sys

# BSP tree
raycaster_env = Tree()

# Flags
ACTION_DELAY_SECONDS = 0.0
ROLLOUTS = 1

DEAD_END_IS_TERMINAL = True

ENERGY_REWARD_WEIGHT = 0.1
TURN_REWARD_WEIGHT = 0.2
TIME_REWARD_WEIGHT = 0.3
REWARD_DEBUG_EVERY = 20

def _fov_to_radians(value): # nice little helper funciton
    if value is None:
        return math.pi / 2
    value = float(value)
    if value > 2 * math.pi:
        value = math.radians(value)
    return min(max(value, 0.0), 2 * math.pi)


def _build_bsp_tree(walls):
    segments = np.asarray(walls, dtype=np.float32)

    if segments.ndim == 2 and segments.shape[1] == 4:
        segments = segments.reshape(-1, 2, 2)

    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    valid = np.isfinite(segments).all(axis=(1, 2)) & (lengths > 1e-6)
    segments = segments[valid]

    p0 = segments[:, 0]
    p1 = segments[:, 1]
    dist_vec = p1 - p0
    segment_vec = p0 + 0.5 * dist_vec
    norm_val = np.linalg.norm(dist_vec, axis=1, keepdims=True)
    normal_vec = np.column_stack((-dist_vec[:, 1], dist_vec[:, 0])) / norm_val
    tree_splitter = raycaster_env.choose_splitter(p0, p1, segment_vec, normal_vec)
    bsp = raycaster_env.create_bsp_tree(p0, p1, segment_vec, normal_vec, tree_splitter)
    return segments, bsp

class ActorCritic(nn.Module):
    def __init__(self, h1, h2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, VARFOV, inputs, action_dim=2):
        super().__init__()
        self.MaxSteps = 50
        self.inputs = int(inputs)
        self.VARNumberOfRayCasts = int(VARNumberOfRayCasts)
        self.VARAllowedEnergy = float(VARAllowedEnergy)
        self.EnergyUsed = 0.0

        self.clear_memory(self.MaxSteps)

        # Actor network
        self.state_norm = nn.LayerNorm(self.inputs)
        self.act1 = nn.Linear(self.inputs, h1)
        self.act2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic network
        self.crit1 = nn.Linear(self.inputs, h1)
        self.crit2 = nn.Linear(h1, h2)
        self.val_out = nn.Linear(h2, 1)

        self.k = nn.Parameter(torch.tensor(5.89))
        self.sd = nn.Parameter(torch.tensor(20.0))

        # Ray offest
        self.RayOffset = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.log_w_turn = nn.Parameter(torch.tensor(math.log(0.05)))
        self.target_max_turn = float(VARtargetMaxTurn)
        self.FOV = _fov_to_radians(VARFOV)
        self.time_cost = 0.0

        self.last_raw_energy = 0.0
        self.last_energy_component = 0.0
        self.last_turn_component = 0.0
        self.last_time_component = 0.0
        self.last_total_penalty = 0.0

    @property
    def w_turn(self):
        return torch.exp(self.log_w_turn)

    @torch.no_grad()
    def clear_memory(self, MaxSteps):
        self.states = torch.zeros((MaxSteps, self.inputs), dtype=torch.float32)
        self.actions = torch.zeros((MaxSteps, 2), dtype=torch.float32)
        self.rewards = torch.zeros(MaxSteps, dtype=torch.float32)
        self.log_probs = torch.zeros(MaxSteps, dtype=torch.float32)
        self.dones = torch.zeros(MaxSteps, dtype=torch.float32)
        self.values = torch.zeros(MaxSteps, dtype=torch.float32)
        self.KoreanMumSpecial = 0.5
        self.BerryGoodKoreanDataHacker = 0.01

    def forward(self, state):
        state_tensor = self.state_norm(torch.as_tensor(state, dtype=torch.float32))

        actor = F.relu(self.act1(state_tensor))
        actor = F.relu(self.act2(actor))
        mean = self.out(actor)
        stds = torch.clamp(torch.exp(self.log_std), min=1e-4, max=5.0)

        critic = F.relu(self.crit1(state_tensor))
        critic = F.relu(self.crit2(critic))
        value = self.val_out(critic).squeeze(-1)
        return mean, stds, value

    def get_action_distribution(self, mean, stds):
        angle_dist = dist.Normal(mean[..., 0], stds[..., 0])
        step_dist = dist.Normal(mean[..., 1], stds[..., 1])
        return angle_dist, step_dist

    @torch.no_grad()
    def get_action_and_value(self, state_list):
        state_tensor = torch.as_tensor(state_list, dtype=torch.float32)
        mean, stds, value = self.forward(state_tensor)
        angle_dist, step_dist = self.get_action_distribution(mean, stds)
        angle_action = angle_dist.sample()
        step_action = step_dist.sample()
        action = torch.stack([angle_action, step_action], dim=-1)
        log_prob = angle_dist.log_prob(angle_action) + step_dist.log_prob(step_action)
        return action.cpu().numpy(), log_prob.cpu(), value.cpu(), mean, stds

    def evaluate_actions(self, states, actions):
        mean, stds, value = self.forward(states)
        angle_dist, step_dist = self.get_action_distribution(mean, stds)
        angles = actions[:, 0]
        step_sizes = actions[:, 1]
        new_log_prob = angle_dist.log_prob(angles) + step_dist.log_prob(step_sizes)
        entropy = angle_dist.entropy() + step_dist.entropy()
        return new_log_prob, entropy, value

    @torch.no_grad()
    def store_transition(self, state, action, reward, log_prob, done, value, training_steps):
        self.states[training_steps] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[training_steps] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[training_steps] = float(reward)
        self.log_probs[training_steps] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.dones[training_steps] = 1.0 if done else 0.0
        self.values[training_steps] = torch.as_tensor(value, dtype=torch.float32)

    @torch.no_grad()
    def InDeadEnd(self, grid_distance, robot_pos, segments):
        if grid_distance <= 0:
            return False, np.asarray(robot_pos, dtype=np.float32)

        region_pos = np.floor(np.asarray(robot_pos) / grid_distance) * grid_distance
        segments = np.asarray(segments, dtype=np.float32)
        if segments.size == 0:
            return False, region_pos
        if segments.ndim == 2 and segments.shape[1] == 4:
            segments = segments.reshape(-1, 2, 2)
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            return False, region_pos

        segment_mids = np.mean(segments, axis=1)
        neighbour_counter = 0
        corners = [
            region_pos,
            region_pos + np.array([0, grid_distance]),
            region_pos + np.array([grid_distance, grid_distance]),
            region_pos + np.array([grid_distance, 0]),
        ]

        for segment, segment_mid in zip(segments, segment_mids):
            seg_p1, seg_p2 = segment
            segment_counted = False
            for point in (seg_p1, seg_p2):
                if segment_counted:
                    break
                if np.allclose(point, corners[0]):
                    if (segment_mid[0] - corners[0][0] > 0) or (segment_mid[1] - corners[0][1] > 0):
                        neighbour_counter += 1
                        segment_counted = True
                        continue
                if np.allclose(point, corners[1]):
                    if (segment_mid[0] - corners[1][0] > 0) or (segment_mid[1] - corners[1][1] < 0):
                        neighbour_counter += 1
                        segment_counted = True
                        continue
                if np.allclose(point, corners[2]):
                    if (segment_mid[0] - corners[2][0] < 0) or (segment_mid[1] - corners[2][1] < 0):
                        neighbour_counter += 1
                        segment_counted = True
                        continue
                if np.allclose(point, corners[3]):
                    if (segment_mid[0] - corners[3][0] < 0) or (segment_mid[1] - corners[3][1] > 0):
                        neighbour_counter += 1
                        segment_counted = True
                        continue

        return neighbour_counter >= 3, region_pos

    @torch.no_grad()
    def EnergyCost(self, angle, step_size, new_angle, done, c1, c2, c3):
        _, energy_value = ConvertRobotAction([new_angle, step_size], done)
        energy_value = float(energy_value)

        # Energy consumption cannot be negative; clamp tiny/invalid negative estimates to zero.
        raw_energy = max(0.0, energy_value)
        angle_diff = abs((new_angle - angle + math.pi) % (2 * math.pi) - math.pi)
        turn_cost = float(self.w_turn.item()) * angle_diff
        compute_time = max(0.0, float(self.time_cost))

        self.last_raw_energy = raw_energy
        self.last_energy_component = float(c1 * raw_energy)
        self.last_turn_component = float(c2 * turn_cost)
        self.last_time_component = float(c3 * compute_time)
        self.last_total_penalty = float(
            self.last_energy_component + self.last_turn_component + self.last_time_component
        )
        return self.last_total_penalty

    def reward(self, run_into_wall, in_dead_end, distance_finish, region_unexplored, robot_pos, new_pos, step_size, success_rate, angle, new_angle, maze_size, done):
        goal_reached = distance_finish < 0.5
        failed_by_collision = bool(run_into_wall)
        failed_by_dead_end = bool(in_dead_end and DEAD_END_IS_TERMINAL)
        action_done = bool(done or failed_by_collision or failed_by_dead_end or goal_reached)

        total_penalty_energy = self.EnergyCost(
            angle, step_size, new_angle, action_done,
            ENERGY_REWARD_WEIGHT, TURN_REWARD_WEIGHT, TIME_REWARD_WEIGHT
        )
        # Energy budget accumulates unweighted energy only, not turn/time penalties.
        self.EnergyUsed += self.last_raw_energy
        safe_maze_size = max(float(maze_size), 1e-6)
        reward = -((1 - success_rate) / 10.0) - 5.0 * (step_size / safe_maze_size) - total_penalty_energy

        if failed_by_collision:
            return float(reward - 10.0), True, False
        # Check success before dead-end status: the goal itself may be in a dead-end cell.
        if goal_reached:
            return float(reward + 100.0), True, True
        if in_dead_end:
            # It always receives the dead-end penalty; whether it terminates is configurable.
            return float(reward - 10.0), DEAD_END_IS_TERMINAL, False
        if self.EnergyUsed > self.VARAllowedEnergy:
            reward -= 1.0
        if region_unexplored:
            reward += 0.1
        if np.allclose(robot_pos, new_pos):
            reward -= 10.0
        return float(reward), False, False

    def InitializeWeights(self, Type):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if Type == "orth":
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                elif Type == "Kaim":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                elif Type == "Xaiv":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.constant_(module.bias, 0.0)

    @staticmethod
    @torch.no_grad()
    def LogDataToCSV(Data1, Data2, Data3, CSVType, agnt_id):
        if CSVType == "eps":
            with open(f"EpisodeData_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data1)
            with open(f"RobotActions_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerows(Data2)
            with open(f"ActionData_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data3)
        elif CSVType == "trg":
            with open(f"TrainingData_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data1)
        else:
            with open(f"Mazes_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([json.dumps(np.asarray(Data1).tolist())])
            with open(f"Accuracies_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([float(Data2)])

    @staticmethod
    @torch.no_grad()
    def GetRobotAccuracy(A_star_path: np.ndarray, Robot_path: np.ndarray):
        A_star_path = np.asarray(A_star_path, dtype=np.float32)
        Robot_path = np.asarray(Robot_path, dtype=np.float32)
        if A_star_path.size == 0 or Robot_path.size == 0:
            return float("inf")
        dtw_distance, alignment_path = fastdtw(A_star_path, Robot_path, dist=euclidean)
        if not alignment_path:
            return float("inf")
        return float(dtw_distance / len(alignment_path))

    @torch.no_grad()
    def GetState(self, robot_pos, VarNumberOfRayCasts, bsp, Rayangle, VARFOV, angle, VARRayDistance, maze_size):
        start_time = time.perf_counter()
        distances = []
        ray_count = int(VarNumberOfRayCasts)

        max_distance = float(maze_size if VARRayDistance is None else VARRayDistance)
        if not math.isfinite(max_distance) or max_distance <= 0:
            max_distance = max(float(maze_size), 1.0)
        ray_offset = float(Rayangle.detach().item()) if torch.is_tensor(Rayangle) else float(Rayangle)
        fov = _fov_to_radians(VARFOV)
        angle_step = (2 * math.pi) / ray_count

        for i in range(ray_count):
            ray_angle = angle_step * i + ray_offset
            angle_difference = abs((ray_angle - float(angle) + math.pi) % (2 * math.pi) - math.pi)
            if angle_difference > fov / 2:
                distances.append(max_distance)
                continue

            ray_vec = np.array([np.cos(ray_angle), np.sin(ray_angle)], dtype=np.float32)
            raw_distance = raycaster_env.traverse_tree(bsp, robot_pos, ray_vec)
            if torch.is_tensor(raw_distance):
                distance = float(raw_distance.detach().cpu().item())
            else:
                distance = float(np.asarray(raw_distance).item())
            if not math.isfinite(distance) or distance >= max_distance:
                distance = max_distance
            distances.append(max(0.0, distance))

        self.time_cost = time.perf_counter() - start_time
        return distances

def StartAgent(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType, VARtargetMaxTurn, agnt, VARFOV, VARRayDistance):
    past = 0
    x = 51
    not_past = 0
    steps = 0
    success_rate = 0.0
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    sum_action_directions = 0.0
    sum_action_step_sizes = 0.0

    network = ActorCritic(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, VARFOV, NNinputs)
    network.InitializeWeights(VARWeightInitType)

    ppo_parameters = [parameter for name, parameter in network.named_parameters() if name != "log_w_turn"]
    optimizer = torch.optim.Adam(ppo_parameters, lr=1e-4)
    w_optimizer = torch.optim.Adam([network.log_w_turn], lr=1e-3)

    dashboard = Dashboard()
    dashboard.start()

    for _rollout in range(ROLLOUTS):
        start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd)
        robot_pos = np.array(start_point, dtype=np.float32)
        end_point = np.array(end_point, dtype=np.float32)
        walls_np, BSP = _build_bsp_tree(walls)
        angle = np.float32((7 * math.pi) / 4)
        A_star_path = Normal_A_star_search(np.array(pixel_grid), list(start_point), list(end_point))
        total_turn_delta = 0.0

        network.clear_memory(network.MaxSteps)
        maze_solved = False
        RegionsExplored = set()
        network.EnergyUsed = 0.0
        robot_path = [robot_pos.copy()]

        last_episode_reward = 0.0
        last_episode_length = 0
        last_maze_solved = False
        last_energy_used = 0.0
        last_regions_explored = 0
        last_accuracy = 0.0
        last_pixel_grid = pixel_grid
        buffer_step = 0
        wall_failures = 0
        dead_end_terminations = 0
        goal_successes = 0
        other_terminations = 0
        lidar_seconds_this_rollout = 0.0

        for step in range(network.MaxSteps):
            done = False
            print(step, end=" ", flush=True)

            State = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size)
            lidar_seconds_this_rollout += network.time_cost
            State.append(float(np.linalg.norm(end_point - robot_pos)))
            State.append(float(maze_size))

            Action, log_prob, value, _, _ = network.get_action_and_value(State)
            dashboard.send(walls_np, np.array([Action[0], Action[1]]))
            sum_action_directions += float(Action[0])
            sum_action_step_sizes += float(Action[1])

            old_angle = float(angle)
            new_angle = math.tanh(float(Action[0])) * math.pi
            step_size = (math.tanh(float(Action[1])) + 1.0) * 2.5
            new_pos = np.floor(robot_pos + step_size * np.array([np.cos(new_angle), np.sin(new_angle)])).astype(np.float32)
            robot_path.append(new_pos.copy())

            angle_diff_step = abs((new_angle - old_angle + math.pi) % (2 * math.pi) - math.pi)
            total_turn_delta += angle_diff_step

            run_into_wall = bool(RunIntoWall(robot_pos, walls, new_pos[0] - robot_pos[0], new_pos[1] - robot_pos[1]))
            in_dead_end, region_pos = network.InDeadEnd(grid_distance, new_pos, walls_np)
            region_tuple = tuple(region_pos.tolist())
            region_unexplored = region_tuple not in RegionsExplored
            if region_unexplored:
                RegionsExplored.add(region_tuple)

            distance_finish = float(np.linalg.norm(end_point - new_pos))
            Reward, done, maze_solved = network.reward(run_into_wall, in_dead_end, distance_finish, region_unexplored, robot_pos, new_pos, step_size, success_rate, old_angle, new_angle, maze_size, done)
            print(done, end=" ",flush=True)
            episode_reward += Reward
            episode_length += 1
            network.store_transition(State, Action, Reward, log_prob, done, value, buffer_step)
            buffer_step += 1
            steps += 1

            completed_actions = past + not_past
            success_rate = 0.0 if completed_actions == 0 else past / completed_actions

            if done:
                last_episode_reward = episode_reward
                last_episode_length = episode_length
                last_maze_solved = maze_solved
                last_energy_used = network.EnergyUsed
                last_regions_explored = len(RegionsExplored)
                last_accuracy = network.GetRobotAccuracy(np.asarray(A_star_path), np.asarray(robot_path))
                print(last_accuracy, end=" ", flush=True)
                last_pixel_grid = pixel_grid

                if maze_solved:
                    past += 1
                    x += 1
                else:
                    not_past += 1

                completed_actions = past + not_past
                success_rate = past / completed_actions if completed_actions else 0.0
                print(success_rate, flush=True)
                network.EnergyUsed = 0.0
                episode_reward = 0.0
                episode_length = 0

                start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd)
                robot_pos = np.array(start_point, dtype=np.float32)
                end_point = np.array(end_point, dtype=np.float32)
                walls_np, BSP = _build_bsp_tree(walls)
                angle = np.float32((7 * math.pi) / 4)
                maze_solved = False
                RegionsExplored = set()
                robot_path = [robot_pos.copy()]
                A_star_path = Normal_A_star_search(np.array(pixel_grid), list(start_point), list(end_point))
            else:
                robot_pos = new_pos
                angle = new_angle

            if ACTION_DELAY_SECONDS > 0:
                time.sleep(ACTION_DELAY_SECONDS)

        if episode_length > 0:
            Episode_Payload = [episode_count, episode_reward, episode_length, 1 if maze_solved else 0, success_rate, network.EnergyUsed, len(RegionsExplored)]
            ACCURACY = network.GetRobotAccuracy(np.asarray(A_star_path), np.asarray(robot_path))
            LoggedPixelGrid = pixel_grid
        else:
            Episode_Payload = [episode_count, last_episode_reward, last_episode_length, 1 if last_maze_solved else 0, success_rate, last_energy_used, last_regions_explored]
            ACCURACY = last_accuracy
            LoggedPixelGrid = last_pixel_grid

        action_payload = [sum_action_directions / network.MaxSteps, sum_action_step_sizes / network.MaxSteps]
        network.LogDataToCSV(Episode_Payload, network.actions.numpy().tolist(), action_payload, "eps", agnt)
        network.LogDataToCSV(LoggedPixelGrid, ACCURACY, None, "mze", agnt)
        episode_count += 1
        episode_reward = 0.0
        network.EnergyUsed = 0.0
        episode_length = 0
        sum_action_directions = 0.0
        sum_action_step_sizes = 0.0

        states = network.states
        actions = network.actions
        rewards = network.rewards
        log_probs = network.log_probs
        dones = network.dones
        values = network.values
        MaxSteps = network.MaxSteps
        Wgamma = 0.99
        Wlambda = 0.99

        if dones[MaxSteps - 1].item() == 1.0:
            next_value = torch.tensor(0.0, dtype=torch.float32)
        else:
            with torch.no_grad():
                FinalState = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size)
                FinalState.append(float(np.linalg.norm(end_point - robot_pos)))
                FinalState.append(float(maze_size))
                _, _, next_value = network.forward(torch.as_tensor(FinalState, dtype=torch.float32))
                next_value = next_value.squeeze(-1)

        advantages = torch.zeros(MaxSteps, dtype=torch.float32)
        next_advantage = torch.tensor(0.0, dtype=torch.float32)
        for t in reversed(range(MaxSteps)):
            next_value_t = next_value if t == MaxSteps - 1 else values[t + 1]
            done_flag = 1.0 - dones[t]
            temp_diff_error = rewards[t] + (Wgamma * next_value_t * done_flag) - values[t]
            advantage = temp_diff_error + (Wgamma * Wlambda) * done_flag * next_advantage
            advantages[t] = advantage
            next_advantage = advantage

        advantages_std, advantages_mean = torch.std_mean(advantages)
        returns = advantages + values
        advantages = (advantages - advantages_mean) / torch.clamp(advantages_std, min=1e-8)

        # TRAINING LOOP
        Start = time.perf_counter()
        PPO_EPOCHS = 4
        params_before_update = [parameter.detach().clone() for parameter in ppo_parameters]
        for epoch in range(PPO_EPOCHS):
            shuffled_indices = torch.randperm(MaxSteps)
            for i in range(0, MaxSteps, 64):
                batch = shuffled_indices[i:i + 64]
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_log_probs = log_probs[batch]
                batch_returns = returns[batch]
                batch_advantages = advantages[batch]

                new_log_prob, entropy, batch_value = network.evaluate_actions(batch_states, batch_actions)
                probability_ratio = torch.exp(new_log_prob - batch_log_probs)
                entropy_mean = entropy.mean()
                surrogate_1 = probability_ratio * batch_advantages
                surrogate_2 = torch.clamp(probability_ratio, 0.8, 1.2) * batch_advantages
                actor_loss = torch.mean(-torch.min(surrogate_1, surrogate_2))
                critic_loss = F.mse_loss(batch_value, batch_returns)
                total_loss = actor_loss + network.BerryGoodKoreanDataHacker * critic_loss - network.KoreanMumSpecial * entropy_mean

                # Update weights!!
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_parameters, max_norm=0.5)
                optimizer.step()

                Training_Payload = [steps, success_rate, total_loss.item(), actor_loss.item(), critic_loss.item(), entropy_mean.item(), batch_advantages.mean().item()]
                network.LogDataToCSV(Training_Payload, None, None, "trg", agnt)

        with torch.no_grad():
            parameter_delta = math.sqrt(sum(float(torch.sum((parameter - before) ** 2).item()) for parameter, before in zip(ppo_parameters, params_before_update)))
        
        avg_turn = total_turn_delta / MaxSteps
        w_turn_loss = -network.w_turn * (avg_turn - network.target_max_turn)
        w_optimizer.zero_grad()
        w_turn_loss.backward()
        w_optimizer.step()
        with torch.no_grad():
            network.log_w_turn.clamp_(math.log(1e-4), math.log(10.0))

        network.clear_memory(MaxSteps)
        print(time.perf_counter() - Start)
        sys.exit()

if __name__ == "__main__":
    StartAgent(48, 24, 24, 100, 26, "orth", 0.5, 0, 360, None)

sys.exit()
