import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from environment.PPO_action_converter import ConvertRobotAction
from environment.maze_utils import raycaster_env, _fov_to_radians

ADAPT_TURN_WEIGHT = False
DEAD_END_IS_TERMINAL = True
ENERGY_REWARD_WEIGHT = 0.1
TURN_REWARD_WEIGHT = 0.2
TIME_REWARD_WEIGHT = 0.0
TURN_EXCESS_WEIGHT = 0.05
MAX_STEP_START = 1.5      
MAX_STEP_END = 5.0        
STEP_RAMP_SUCCESS = 0.4

REWARD_CLIP = 1.0
STEP_TIME_COST = 0.01       
CLEARANCE_WEIGHT = 0.05    
CLEARANCE_DISTANCE = 1.5
VISIT_PENALTY = 0.02        
VISIT_CAP = 5             
EXPLORE_BONUS = 0.1       
STALL_STEP_SIZE = 0.25     
STALL_STEP_PENALTY = 0.2   
OVER_BUDGET_PENALTY = 0.2 
RAY_SCALE = 24.0
LOG_STD_MIN = -2.0
HARD_ENERGY_LIMIT = True

class ActorCritic(nn.Module):
    def __init__(self, h1, h2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, VARFOV, inputs, action_dim=2):
        super().__init__()
        self.MaxSteps = 100
        self.inputs = int(inputs)
        self.VARNumberOfRayCasts = int(VARNumberOfRayCasts)
        self.VARAllowedEnergy = float(VARAllowedEnergy)
        
        self.EnergyUsed = 0.0
        self.last_action = [0.0, 0.0]
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

        # added a way for the AI to optimize its cirriculum
        self.k = nn.Parameter(torch.tensor(5.89), requires_grad=False)
        self.sd = nn.Parameter(torch.tensor(20.0), requires_grad=False)

        # Ray offset
        self.RayOffset = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=False)

        # other optimizable parameters
        self.log_w_turn = nn.Parameter(torch.tensor(math.log(0.05)), requires_grad=ADAPT_TURN_WEIGHT)

        self.target_max_turn = float(VARtargetMaxTurn)
        self.FOV = _fov_to_radians(VARFOV)
        self.max_step = MAX_STEP_START
        self.step_scale = MAX_STEP_START / 2.0 
        self.last_rays_cast = 0
        self.last_raw_energy = 0.0
        self.last_energy_component = 0.0
        self.last_turn_component = 0.0
        self.last_time_component = 0.0
        self.last_total_penalty = 0.0

    @property
    def w_turn(self):
        return torch.exp(self.log_w_turn)

    def set_max_step(self, success_rate):
        # Only call this between PPO updates (buffer empty): the stored log-probs use the scale they were collected with.
        self.max_step = MAX_STEP_START + (MAX_STEP_END - MAX_STEP_START) * min(1.0, success_rate / STEP_RAMP_SUCCESS)
        self.step_scale = self.max_step / 2.0

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
        stds = self.log_std.clamp(LOG_STD_MIN, 1.0).exp()
        critic = torch.tanh(self.crit1(state_tensor))
        critic = torch.tanh(self.crit2(critic))
        value = self.val_out(critic).squeeze(-1)
        return mean, stds, value

    def get_action_distribution(self, mean, stds):
        return dist.Normal(mean[..., 0], stds[..., 0]), dist.Normal(mean[..., 1], stds[..., 1])

    def _squash_action(self, latent_action):
        bounded = torch.tanh(latent_action)
        return torch.stack([math.pi * bounded[..., 0], self.step_scale * (bounded[..., 1] + 1.0)], dim=-1)

    def _action_log_prob(self, mean, stds, latent_action):
        # uses jacobian
        log_jacobian = 2.0 * (math.log(2.0) - latent_action - F.softplus(-2.0 * latent_action))
        log_jacobian = log_jacobian + latent_action.new_tensor([math.log(math.pi), math.log(self.step_scale)])
        return (dist.Normal(mean, stds).log_prob(latent_action) - log_jacobian).sum(dim=-1)

    @torch.no_grad()
    def get_action_and_value(self, state_list):
        mean, stds, value = self.forward(state_list)
        latent_action = dist.Normal(mean, stds).sample()
        action = self._squash_action(latent_action)
        log_prob = self._action_log_prob(mean, stds, latent_action)

        return action.cpu().numpy(), log_prob.cpu(), value.cpu(), latent_action.cpu(), stds

    def evaluate_actions(self, states, actions):
        mean, stds, value = self.forward(states)
        new_log_prob = self._action_log_prob(mean, stds, actions)

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

        pieces = []
        for seg_a, seg_b in segments:
            axis = 1 if seg_a[0] == seg_b[0] else 0
            lo, hi = sorted((float(seg_a[axis]), float(seg_b[axis])))
            cuts = [lo] + [float(k * grid_distance) for k in range(int(lo // grid_distance) + 1, int(np.ceil(hi / grid_distance))) if lo < k * grid_distance < hi] + [hi]
            for u, v in zip(cuts[:-1], cuts[1:]):
                p1, p2 = seg_a.copy(), seg_b.copy()
                p1[axis], p2[axis] = u, v
                pieces.append([p1, p2])
        segments = np.asarray(pieces, dtype=np.float32)

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
    def EnergyCost(self, angle, step_size, new_angle, done, c1, c2, c3, VARRayDistance, maze_size):
        turn = (new_angle - angle + math.pi) % (2 * math.pi) - math.pi
        _, energy_value = ConvertRobotAction([turn, step_size], done)
        energy_value = float(energy_value)
        raw_energy = max(0.0, energy_value)
        angle_diff = abs((new_angle - angle + math.pi) % (2 * math.pi) - math.pi)
        turn_cost = float(self.w_turn.item()) * angle_diff

        max_distance = float(maze_size if VARRayDistance is None else VARRayDistance)
        compute_time = float(0.002 * self.last_rays_cast * (max_distance / 10) ** 1)
        self.last_raw_energy = raw_energy

        # normalised by the maze size, like the progress reward, so the energy penalty is a fixed fraction of what a
        # straight step toward the goal earns (and does not change when the energy budget is changed)

        self.last_energy_component = float(c1 * raw_energy / maze_size)
        self.last_turn_component = float(c2 * turn_cost)
        self.last_time_component = float(c3 * compute_time)
        self.last_total_penalty = float(
            self.last_energy_component + self.last_turn_component + self.last_time_component
        )
        return self.last_total_penalty

    def reward(self, distance_progress, run_into_wall, in_dead_end, distance_finish, region_unexplored, VARRayDistance, new_pos, step_size, success_rate, angle, new_angle, maze_size, done, min_ray_px, turn_excess=0.0, revisits=0):
        goal_reached = distance_finish < 0.5

        failed_by_collision = bool(run_into_wall)
        failed_by_dead_end = bool(in_dead_end and DEAD_END_IS_TERMINAL)

        action_done = bool(done or failed_by_collision or failed_by_dead_end or goal_reached)

        total_penalty_energy = self.EnergyCost(
            angle, step_size, new_angle, action_done,
            ENERGY_REWARD_WEIGHT * min(1.0, success_rate / 0.4), TURN_REWARD_WEIGHT, TIME_REWARD_WEIGHT, VARRayDistance, maze_size
        )

        excess_penalty = TURN_EXCESS_WEIGHT * turn_excess
        self.last_turn_component += excess_penalty
        self.last_total_penalty += excess_penalty
        total_penalty_energy += excess_penalty

        self.EnergyUsed += self.last_raw_energy
        safe_maze_size = max(float(maze_size), 1e-6)
        reward = (1.0 * (distance_progress / safe_maze_size) - STEP_TIME_COST - total_penalty_energy
                  - CLEARANCE_WEIGHT * max(0.0, 1.0 - min_ray_px / CLEARANCE_DISTANCE)
                  - VISIT_PENALTY * min(revisits, VISIT_CAP))
        reward = max(-REWARD_CLIP, min(REWARD_CLIP, reward))
        
        if failed_by_collision:
            return float(reward - 5.0), True, False
        if goal_reached:
            return float(reward + 10.0), True, True
        if in_dead_end:
            return float(reward - 5.0), DEAD_END_IS_TERMINAL, False
        if self.EnergyUsed > self.VARAllowedEnergy:
            if HARD_ENERGY_LIMIT:
                return float(reward - 5.0), True, False
            reward -= OVER_BUDGET_PENALTY
        if region_unexplored:
            reward += EXPLORE_BONUS
        if step_size < STALL_STEP_SIZE:
            reward -= STALL_STEP_PENALTY
        
        return float(max(-REWARD_CLIP, min(REWARD_CLIP, reward))), False, False

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

        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.orthogonal_(self.val_out.weight, gain=1.0)

    @staticmethod
    @torch.no_grad()
    def LogDataToCSV(Data1, Data2, Data3, CSVType, agnt_id, directory=".", test=""):
        if CSVType == "ep":
            with open(Path(directory) / f"EpisodeData_{test}{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data1)
        elif CSVType == "act":
            with open(Path(directory) / f"ActionData_{test}{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data2)
        elif CSVType == "trg":
            with open(Path(directory) / f"TrainingData_{test}{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(Data1)
        elif CSVType == "mze":
            with open(Path(directory) / f"Mazes_{test}{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([json.dumps(np.asarray(Data1).tolist())])
            with open(Path(directory) / f"Accuracies_{test}{agnt_id}.csv", mode="a", newline="", encoding="utf-8") as file:
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
    
    @staticmethod
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    
    @torch.no_grad()
    def GetState(self, robot_pos, VarNumberOfRayCasts, bsp, Rayangle, VARFOV, angle, VARRayDistance, maze_size, end_point):
        distances = []
        self.last_rays_cast = 0
        max_ray_angle = 0
        max_ray_distance = 0
        ray_count = int(VarNumberOfRayCasts)
        max_distance = float(maze_size if VARRayDistance is None else VARRayDistance)
        if not math.isfinite(max_distance) or max_distance <= 0:
            max_distance = max(float(maze_size), 1.0)
        ray_offset = float(Rayangle.detach().item()) if torch.is_tensor(Rayangle) else float(Rayangle)
        fov = _fov_to_radians(VARFOV)

        if ray_count == 1:
            relative_angles = [0.0]
        elif fov >= 2 * math.pi - 1e-6:
            relative_angles = [(2 * math.pi) * i / ray_count - math.pi for i in range(ray_count)]
        else:
            relative_angles = [fov * (i / (ray_count - 1) - 0.5) for i in range(ray_count)]
        for i in range(ray_count):
            ray_angle = float(angle) + relative_angles[i] + ray_offset
            ray_vec = np.array([np.cos(ray_angle), np.sin(ray_angle)], dtype=np.float32)
            self.last_rays_cast += 1
            raw_distance = raycaster_env.traverse_tree(bsp, robot_pos, ray_vec)

            if torch.is_tensor(raw_distance):
                distance = float(raw_distance.detach().cpu().item())
            else:
                distance = float(np.asarray(raw_distance).item())
            if not math.isfinite(distance) or distance >= max_distance:
                distance = max_distance
            final_distance = max(0.0, distance)
            distances.append(final_distance)

            if final_distance > max_ray_distance:
               max_ray_angle = ray_angle
               max_ray_distance = final_distance
            
        energy_remaining = max(0.0, 1.0 - self.EnergyUsed / self.VARAllowedEnergy)
        angle_finish = self.wrap(math.atan2(end_point[1] - robot_pos[1], end_point[0] - robot_pos[0]) - float(angle))
        distance_finish = float(np.linalg.norm(end_point - robot_pos)) / RAY_SCALE

        return [math.sqrt(min(1.0, d / RAY_SCALE)) for d in distances] + [
            energy_remaining, math.sin(angle_finish), math.cos(angle_finish), distance_finish,
            float(self.last_action[0]), float(self.last_action[1]),
        ], max_ray_angle