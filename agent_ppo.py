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
import random
from pathlib import Path

# BSP tree
raycaster_env = Tree()

# Flags
ACTION_DELAY_SECONDS = 0.0
ROLLOUTS = 5
SEED = 42
MIN_ROLLOUT_STEPS = 200 # must be positive
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
TARGET_KL = 0.02
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
ADAPT_TURN_WEIGHT = False
CURRICULUM_ENABLED = False
DEAD_END_IS_TERMINAL = True
ENERGY_REWARD_WEIGHT = 0.1
TURN_REWARD_WEIGHT = 0.2
TIME_REWARD_WEIGHT = 0.0  
REWARD_DEBUG_EVERY = 20

def _fov_to_radians(value):
    if value is None:
        return math.pi / 2
    value = float(value)
    if value > 2 * math.pi:
        value = math.radians(value)
    return min(max(value, 0.0), 2 * math.pi)

def _build_bsp_tree(walls):
    segments = np.asarray(walls, dtype=np.float32)
    if segments.size == 0:
        segments = segments.reshape(0, 2, 2)
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
        self.MaxSteps = 100
        self.inputs = int(inputs)
        self.VARNumberOfRayCasts = int(VARNumberOfRayCasts)
        self.VARAllowedEnergy = float(VARAllowedEnergy)
        if action_dim != 2 or self.VARNumberOfRayCasts <= 0:
            raise ValueError("Use two actions and at least one LiDAR ray.")
        if self.inputs != self.VARNumberOfRayCasts + 5:
            raise ValueError("NNinputs must equal VARNumberOfRayCasts + 5 (goal, heading, budget).")
        if not math.isfinite(self.VARAllowedEnergy) or self.VARAllowedEnergy <= 0:
            raise ValueError("VARAllowedEnergy must be finite and positive.")
        self.EnergyUsed = 0.0
        self.clear_memory(self.MaxSteps)
        # Actor network 
        self.act1 = nn.Linear(self.inputs, h1)
        self.act2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        # Critic network
        self.crit1 = nn.Linear(self.inputs, h1)
        self.crit2 = nn.Linear(h1, h2)
        self.val_out = nn.Linear(h2, 1)
        # Fixed experiment settings: the imported simulator is not differentiable.
        self.k = nn.Parameter(torch.tensor(5.89), requires_grad=False)
        self.sd = nn.Parameter(torch.tensor(20.0), requires_grad=False)
        # Ray offset
        self.RayOffset = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=False)
        self.log_w_turn = nn.Parameter(torch.tensor(math.log(0.05)), requires_grad=ADAPT_TURN_WEIGHT)
        self.target_max_turn = float(VARtargetMaxTurn)
        self.FOV = _fov_to_radians(VARFOV)
        self.time_cost = 0.0
        self.last_rays_cast = 0
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
    def forward(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if state_tensor.shape[-1] != self.inputs or not torch.isfinite(state_tensor).all():
            raise ValueError("State must have the configured width and finite values.")
        actor = torch.tanh(self.act1(state_tensor))
        actor = torch.tanh(self.act2(actor))
        mean = self.out(actor)
        stds = self.log_std.clamp(-3.0, 1.0).exp()
        critic = torch.tanh(self.crit1(state_tensor))
        critic = torch.tanh(self.crit2(critic))
        value = self.val_out(critic).squeeze(-1)
        return mean, stds, value

    def get_action_distribution(self, mean, stds):
        return dist.Normal(mean[..., 0], stds[..., 0]), dist.Normal(mean[..., 1], stds[..., 1])

    @staticmethod
    def _squash_action(latent_action):
        bounded = torch.tanh(latent_action)
        return torch.stack([math.pi * bounded[..., 0], 2.5 * (bounded[..., 1] + 1.0)], dim=-1)

    @staticmethod
    def _action_log_prob(mean, stds, latent_action):
        # Stable log derivative of tanh, including angle/step scaling.
        log_jacobian = 2.0 * (math.log(2.0) - latent_action - F.softplus(-2.0 * latent_action))
        log_jacobian = log_jacobian + latent_action.new_tensor([math.log(math.pi), math.log(2.5)])
        return (dist.Normal(mean, stds).log_prob(latent_action) - log_jacobian).sum(dim=-1)

    @torch.no_grad()
    def get_action_and_value(self, state_list):
        mean, stds, value = self.forward(state_list)
        latent_action = dist.Normal(mean, stds).sample()
        action = self._squash_action(latent_action)
        log_prob = self._action_log_prob(mean, stds, latent_action)
        # Fourth return value is the latent sample; keep it for exact PPO scoring.
        return action.cpu().numpy(), log_prob.cpu(), value.cpu(), latent_action.cpu(), stds

    def evaluate_actions(self, states, actions):
        # actions contains stored PRE-tanh samples, not physical commands.
        mean, stds, value = self.forward(states)
        new_log_prob = self._action_log_prob(mean, stds, actions)
        # Reparameterized Monte Carlo entropy of the BOUNDED action distribution.
        sample = dist.Normal(mean, stds).rsample()
        entropy = -self._action_log_prob(mean, stds, sample)
        return new_log_prob, entropy, value

    @torch.no_grad()
    def store_transition(self, state, action, reward, log_prob, done, value, training_steps):
        if not 0 <= training_steps < len(self.rewards):
            raise IndexError("Rollout buffer capacity exceeded.")
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
        if not math.isfinite(energy_value):
            raise ValueError("ConvertRobotAction returned non-finite energy.")
        raw_energy = max(0.0, energy_value)
        angle_diff = abs((new_angle - angle + math.pi) % (2 * math.pi) - math.pi)
        turn_cost = float(self.w_turn.item()) * angle_diff
        compute_time = max(0.0, float(self.time_cost))
        self.last_raw_energy = raw_energy
        # One unit is the episode energy budget divided by the step limit.
        energy_scale = self.VARAllowedEnergy / self.MaxSteps
        self.last_energy_component = float(c1 * raw_energy / energy_scale)
        self.last_turn_component = float(c2 * turn_cost)
        self.last_time_component = float(c3 * compute_time)
        self.last_total_penalty = float(
            self.last_energy_component + self.last_turn_component + self.last_time_component
        )
        return self.last_total_penalty
    def reward(self, distance_progress, run_into_wall, in_dead_end, distance_finish, region_unexplored, robot_pos, new_pos, step_size, success_rate, angle, new_angle, maze_size, done):
        goal_reached = distance_finish < 0.5
        failed_by_collision = bool(run_into_wall)
        failed_by_dead_end = bool(in_dead_end and DEAD_END_IS_TERMINAL)
        action_done = bool(done or failed_by_collision or failed_by_dead_end or goal_reached)
        total_penalty_energy = self.EnergyCost(
            angle, step_size, new_angle, action_done,
            ENERGY_REWARD_WEIGHT, TURN_REWARD_WEIGHT, TIME_REWARD_WEIGHT
        )
        self.EnergyUsed += self.last_raw_energy
        safe_maze_size = max(float(maze_size), 1e-6)
        reward = 1.0 * (distance_progress / safe_maze_size) - 0.1 * (step_size / safe_maze_size) - total_penalty_energy 
        if failed_by_collision:
            return float(reward - 5.0), True, False
        if goal_reached:
            return float(reward + 10.0), True, True
        if in_dead_end:
            return float(reward - 5.0), DEAD_END_IS_TERMINAL, False
        if self.EnergyUsed > self.VARAllowedEnergy:
            reward -= 1.0
        if region_unexplored:
            reward += 0.1
        if np.allclose(robot_pos, new_pos):
            reward -= 2.0
        return float(reward), False, False
    def InitializeWeights(self, Type):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if Type == "orth":
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                elif Type == "Kaim":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="tanh")
                elif Type == "Xaiv":
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("tanh"))
                else:
                    raise ValueError("Weight initialization must be 'orth', 'Kaim', or 'Xaiv'.")
                nn.init.zeros_(module.bias)
        # Start with small actor means, independent of observation width.
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.orthogonal_(self.val_out.weight, gain=1.0)

    @staticmethod
    @torch.no_grad()
    def LogDataToCSV(Data1, Data2, Data3, CSVType, agnt_id, directory="."):
        if CSVType == "ep":
            with open(Path(directory) / f"EpisodeData_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data1)
        elif CSVType == "act":
            with open(Path(directory) / f"ActionData_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data2)
        elif CSVType == "trg":
            with open(Path(directory) / f"TrainingData_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data1)
        elif CSVType == "mze":
            with open(Path(directory) / f"Mazes_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([json.dumps(np.asarray(Data1).tolist())])
            with open(Path(directory) / f"Accuracies_{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
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
    def GetState(self, robot_pos, VarNumberOfRayCasts, bsp, Rayangle, VARFOV, angle, VARRayDistance, maze_size, end_point):
        start_time = time.perf_counter()
        distances = []
        self.last_rays_cast = 0
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
            self.last_rays_cast += 1
            raw_distance = raycaster_env.traverse_tree(bsp, robot_pos, ray_vec)

            if torch.is_tensor(raw_distance):
                distance = float(raw_distance.detach().cpu().item())
            else:
                distance = float(np.asarray(raw_distance).item())
            if not math.isfinite(distance) or distance >= max_distance:
                distance = max_distance
            distances.append(max(0.0, distance))

        self.time_cost = time.perf_counter() - start_time

        goal_delta = (np.asarray(end_point) - np.asarray(robot_pos)) / max(float(maze_size), 1e-6)
        energy_remaining = max(0.0, 1.0 - self.EnergyUsed / self.VARAllowedEnergy)

        return [d / max_distance for d in distances] + [
            float(goal_delta[0]), float(goal_delta[1]),
            math.sin(float(angle)), math.cos(float(angle)), energy_remaining,
        ]

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
def StartAgent(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType, VARtargetMaxTurn, agnt, VARFOV, VARRayDistance, *, seed=SEED, rollouts=None, output_dir=None):
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
    
    dashboard = Dashboard()
    dashboard.start()

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
        with open(directory / f"{name}_{agnt}.csv", "x", newline="", encoding="utf-8") as f:
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

        # Initializations
        angle = float((7 * math.pi) / 4)
        A_star_path = Normal_A_star_search(np.array(pixel_grid), list(start_point), list(end_point))
        total_turn_delta = 0.0
        episode_start = buffered_steps
        RegionsExplored = set()

        if grid_distance > 0:
            RegionsExplored.add(tuple((np.floor(robot_pos / grid_distance) * grid_distance).tolist()))

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
            Dashboard.send(np.array(Action), walls_np)

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
            region_unexplored = region_tuple not in RegionsExplored
            if region_unexplored:
                RegionsExplored.add(region_tuple)

            # Get reward 
            next_distance_finish = float(np.linalg.norm(end_point - new_pos))
            distance_progress = distance_finish - next_distance_finish
            Reward, done, maze_solved = network.reward(
                distance_progress, run_into_wall, in_dead_end, next_distance_finish,
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
                                       network.last_time_component], None, "act", agnt, directory)
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
        network.LogDataToCSV(Episode_Payload, None, None, "ep", agnt, directory)
        network.LogDataToCSV(pixel_grid, ACCURACY, None, "mze", agnt, directory)
        episode_count += 1

        print( # standard debug I think
            f"Episode {episode_count}/{rollouts} | "
            f"Steps: {episode_length} | "
            f"Reward: {episode_reward:.3f} | "
            f"Success: {maze_solved} | "
            f"Termination: {termination}"
        )

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
            network.LogDataToCSV([episode_count, success_rate] + row, None, None, "trg", agnt, directory)

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

if __name__ == "__main__":
    final_network = StartAgent(48, 24, 24, 9000, 29, "orth", 0.5, 0, 360, None)