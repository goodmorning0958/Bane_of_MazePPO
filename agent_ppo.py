import csv
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from Bane_of_mazePPO.training_example_generator import get_new_training_example 
from Bane_of_mazePPO.agent_viewer import Dashboard 
from Bane_of_mazePPO.simulated_LiDAR import LiDAR 
from Bane_of_mazePPO.A_star_search import Normal_A_star_search

raycaster_env = LiDAR() 


class ActorCritic(nn.Module): 
    def __init__(self, h1, h2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, inputs, action_dim=2): 
        super().__init__() 
        self.MaxSteps = 200 
        self.inputs = inputs
        self.VARNumberOfRayCasts = VARNumberOfRayCasts 
        self.VARAllowedEnergy = VARAllowedEnergy 
        self.EnergyUsed = 0 

        self.clear_memory(self.MaxSteps) 
        
        # Actor Network  
        self.act1 = nn.Linear(inputs, h1) 
        self.act2 = nn.Linear(h1, h2) 
        self.out = nn.Linear(h2, action_dim) 
        self.log_std = nn.Parameter(torch.zeros(action_dim)) 
        
        # Critic Network  
        self.crit1 = nn.Linear(inputs, h1) 
        self.crit2 = nn.Linear(h1, h2) 
        self.val_out = nn.Linear(h2, 1) 

        self.k = nn.Parameter(torch.tensor(5.89)) 
        self.sd = nn.Parameter(torch.tensor(20.0)) 
        
        # Lagrangian constraints for Turn Energy
        self.log_w_turn = nn.Parameter(torch.tensor(math.log(0.05)))  
        self.target_max_turn = VARtargetMaxTurn

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
        # Actor 
        a = F.relu(self.act1(state)) 
        a = F.relu(self.act2(a)) 
        mean = self.out(a) 
        stds = torch.exp(self.log_std) 
        
        # Critic 
        c = F.relu(self.crit1(state)) 
        c = F.relu(self.crit2(c)) 
        value = self.val_out(c).squeeze(-1) 
        
        return mean, stds, value 

    def get_action_distribution(self, mean, stds):
        angle_dist = dist.TransformedDistribution(
            dist.Normal(mean[..., 0], stds[..., 0]),
            [
                dist.TanhTransform(),
                dist.AffineTransform(loc=0.0, scale=math.pi)
            ]
        )
        step_dist = dist.Normal(mean[..., 1], stds[..., 1])
        return angle_dist, step_dist

    @torch.no_grad()
    def get_action_and_value(self, state_list):
        state_tensor = torch.FloatTensor(state_list)
        mean, stds, value = self.forward(state_tensor)
        angle_dist, step_dist = self.get_action_distribution(mean, stds)

        angle = angle_dist.sample()
        step_size = step_dist.sample()

        action = torch.stack([angle, step_size], dim=-1)
        log_prob = angle_dist.log_prob(angle) + step_dist.log_prob(step_size)

        return action.numpy(), log_prob, value, mean, stds

    def evaluate_actions(self, states, actions):
        mean, stds, value = self.forward(states)
        angle_dist, step_dist = self.get_action_distribution(mean, stds)

        angles = actions[:, 0]
        step_sizes = actions[:, 1]

        angle_log_prob = angle_dist.log_prob(angles)
        step_log_prob = step_dist.log_prob(step_sizes)

        new_log_prob = angle_log_prob + step_log_prob

        angle_entropy = dist.Normal(mean[:, 0], stds[:, 0]).entropy()
        step_entropy = step_dist.entropy()
        entropy = angle_entropy + step_entropy

        return new_log_prob, entropy, value
    
    @torch.no_grad() 
    def store_transition(self, state, action, reward, log_prob, done, value, training_steps): 
        self.states[training_steps] = torch.FloatTensor(state) 
        self.actions[training_steps] = torch.FloatTensor(action) 
        self.rewards[training_steps] = reward 
        self.log_probs[training_steps] = log_prob 
        self.dones[training_steps] = 1.0 if done else 0.0 
        self.values[training_steps] = value 

    @torch.no_grad() 
    def InDeadEnd(self, grid_distance, robot_pos, segments): 
        region_pos = np.floor(robot_pos / grid_distance) * grid_distance 
        segment_mids = np.mean(segments, axis=1) 
        neibour_counter = 0 

        corners = [ 
            region_pos,                               
            region_pos + np.array([0, grid_distance]),         
            region_pos + np.array([grid_distance, grid_distance]),  
            region_pos + np.array([grid_distance, 0])          
        ] 

        for segment, segment_mid in zip(segments.tolist(), segment_mids.tolist()): 
            seg_p1, seg_p2 = np.array(segment[0]), np.array(segment[1]) 
            segment_counted = False   

            for p in [seg_p1, seg_p2]: 
                if segment_counted: 
                    break 
                
                if np.allclose(p, corners[0]):  
                    if (segment_mid[0] - corners[0][0] > 0) or (segment_mid[1] - corners[0][1] > 0): 
                        neibour_counter += 1; segment_counted = True; continue 
                if np.allclose(p, corners[1]): 
                    if (segment_mid[0] - corners[1][0] > 0) or (segment_mid[1] - corners[1][1] < 0): 
                        neibour_counter += 1; segment_counted = True; continue 
                if np.allclose(p, corners[2]): 
                    if (segment_mid[0] - corners[2][0] < 0) or (segment_mid[1] - corners[2][1] < 0): 
                        neibour_counter += 1; segment_counted = True; continue 
                if np.allclose(p, corners[3]): 
                    if (segment_mid[0] - corners[3][0] < 0) or (segment_mid[1] - corners[3][1] > 0): 
                        neibour_counter += 1; segment_counted = True; continue 

        return (neibour_counter >= 3), region_pos 
    
    @torch.no_grad() 
    def RunnIntoWall(self, walls, new_pos, robot_pos): 
        dx = new_pos[0] - robot_pos[0] 
        dy = new_pos[1] - robot_pos[1] 
        t_distance = raycaster_env.CastRay(robot_pos, walls, dx, dy) 
        return t_distance <= 1.0 
        
    def reward(self, run_into_wall, in_dead_end, distance_finish, region_unexplored, robot_pos, new_pos, step_size, sucess_rate, robot_angle, new_angle): 
        Times = {32: 0.0013145, 16: 0.0005158, 8: 0.0003281, 4: 0.0002455} 
        time_cost = Times.get(self.VARNumberOfRayCasts, 0.001) 
        
        step_size = max(abs(step_size), 1e-5) 
        angle_diff = (new_angle - robot_angle + math.pi) % (2 * math.pi) - math.pi 
        turn_penalty = self.w_turn.item() * abs(angle_diff) 
        
        reward = - (0.01 + time_cost + turn_penalty) - ((1 - sucess_rate) / 10.0) 
        self.EnergyUsed += time_cost 

        if run_into_wall:  
            return reward - 10.0, True, False 

        if in_dead_end:  
            return reward - 10.0, False, False 
        
        if distance_finish < 0.5: 
            return reward + 100.0, True, True 

        if self.EnergyUsed > self.VARAllowedEnergy: 
            reward -= 1.0 

        if region_unexplored: 
            reward += 0.1 

        if np.allclose(robot_pos, new_pos): 
            reward -= 10.0 

        return reward, False, False 

    def InitializeWeights(self, Type):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if Type == "orth":
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif Type == "Kaim":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif Type == "Xaiv":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    raise TypeError(f"Unknown initialization type: {Type}")
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    @torch.no_grad() 
    def LogDataToCSV(Data1, Data2, Data3, CSVType, agnt_id): 
        if CSVType == 'eps': 
            with open(f'EpisodeData_{agnt_id}.csv', mode='a', newline='') as file: 
                csv.writer(file).writerow(Data1) 
            with open(f'RobotActions_{agnt_id}.csv', mode='a', newline='') as file: 
                csv.writer(file).writerow(Data2) 
            with open(f'ActionData_{agnt_id}.csv', mode='a', newline='') as file: 
                csv.writer(file).writerow(Data3) 
        elif CSVType == 'trg':
            with open(f'TrainingData_{agnt_id}.csv', mode='a', newline='') as file: 
                csv.writer(file).writerow(Data1) 
        else:
            with open(f'Mazes_{agnt_id}.csv', mode='a', newline='') as file:
                csv.writer(file).writerow(Data1)
            with open(f'Accuracies_{agnt_id}.csv', mode='a', newline='') as file:
                csv.writer(file).writerow(Data2)

    @staticmethod
    @torch.no_grad()             
    def GetRobotAccuracy(A_star_path: np.array, Robot_path: np.array):
        dtw_distance, alignment_path = fastdtw(A_star_path, Robot_path, dist=euclidean)
        return dtw_distance / len(alignment_path)


@torch.no_grad() 
def GetState(robot_pos, walls):   
    distances = [] 
    angle_step = (2 * math.pi) / 32 
    for dir_ in range(32): 
        distances.append(raycaster_env.CastRay(np.asarray(robot_pos), walls, math.cos(dir_ * angle_step), math.sin(dir_ * angle_step))) 
    return distances 


def StartAgent(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType, VARtargetMaxTurn, agnt): 
    past = 0 
    x = 0 
    not_past = 0 
    steps = 0 
    sucess_rate = 1.0 

    episode_count = 0 
    episode_reward = 0 
    episode_length = 0 
    sum_action_directions = 0 
    sum_action_steps_sizes = 0 

    network = ActorCritic(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, NNinputs) 
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4) 
    w_optimizer = torch.optim.Adam([network.log_w_turn], lr=1e-3)
    
    dashboard = Dashboard() 
    dashboard.start() 

    network.InitializeWeights(VARWeightInitType) 

    while True: 
        start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd) 
        robot_pos = np.array(start_point, dtype=np.float32) 
        end_point = np.array(end_point, dtype=np.float32) 
        walls_np = np.array(walls) 
        maze_np = np.array(pixel_grid) 
        robot_angle = 0.0 

        A_star_path = Normal_A_star_search(pixel_grid.tolist())
        total_turn_delta = 0.0

        network.clear_memory(network.MaxSteps)  
        maze_solved = False 

        RegionsExplored = set()
        network.EnergyUsed = 0

        robot_path = [robot_pos.copy()]

        last_episode_reward = 0
        last_episode_length = 0
        last_maze_solved = False
        last_energy_used = 0
        last_regions_explored = 0
        last_accuracy = 0
        last_pixel_grid = pixel_grid
        
        for step in range(network.MaxSteps):  
            State = GetState(robot_pos, walls)   
            State.append(float(np.linalg.norm(end_point - robot_pos))) 
            State.append(float(maze_size)) 
            
            Action, log_prob, value, _, _ = network.get_action_and_value(State) 
            sum_action_directions += Action[0] 
            sum_action_steps_sizes += Action[1] 

            angle, step_size = Action[0], Action[1]  
            new_pos = robot_pos + step_size * np.array([np.cos(angle), np.sin(angle)]) 
            new_angle = angle 

            robot_path.append(new_pos.copy())
            
            angle_diff_step = abs((new_angle - robot_angle + math.pi) % (2 * math.pi) - math.pi)
            total_turn_delta += angle_diff_step

            run_into_wall = network.RunnIntoWall(walls, new_pos, robot_pos) 
            in_dead_end, region_pos = network.InDeadEnd(grid_distance, new_pos, walls_np) 
        
            region_tuple = tuple(region_pos.tolist()) 
            region_unexplored = region_tuple not in RegionsExplored 
            if region_unexplored: 
                RegionsExplored.add(region_tuple) 

            distance_finish = np.linalg.norm(end_point - new_pos) 
            Reward, done, maze_solved = network.reward(run_into_wall, in_dead_end, distance_finish, region_unexplored, robot_pos, new_pos, step_size, sucess_rate, robot_angle, new_angle) 
            episode_reward += Reward 
            episode_length += 1

            network.store_transition(State, Action, Reward, log_prob, done, value, step) 
            dashboard.send(maze_np.tolist(), [[new_pos[0], new_pos[1], new_angle]]) 

            steps += 1 
            completed_actions = past + not_past
            sucess_rate = 1.0 if completed_actions == 0 else past / completed_actions 
        
            if done: 
                last_episode_reward = episode_reward
                last_episode_length = episode_length
                last_maze_solved = maze_solved
                last_energy_used = network.EnergyUsed
                last_regions_explored = len(RegionsExplored)
                last_accuracy = network.GetRobotAccuracy(np.asarray(robot_path), np.asarray(A_star_path))
                last_pixel_grid = pixel_grid

                if maze_solved: 
                    past += 1 
                    x += 1 
                else: 
                    not_past += 1 

                network.EnergyUsed = 0
                episode_reward = 0
                episode_length = 0

                start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd) 
                robot_pos = np.array(start_point, dtype=np.float32) 
                end_point = np.array(end_point, dtype=np.float32) 
                walls_np = np.array(walls) 
                maze_np = np.array(pixel_grid) 
                robot_angle = 0.0 
                maze_solved = False
                RegionsExplored = set()
                robot_path = [robot_pos.copy()]
                A_star_path = Normal_A_star_search(pixel_grid.tolist())
            else: 
                robot_pos = new_pos 
                robot_angle = new_angle 

        if episode_length > 0:
            Episode_Payload = [episode_count, episode_reward, episode_length, 1 if maze_solved else 0, sucess_rate, network.EnergyUsed, len(RegionsExplored)]
            ACCURACY = network.GetRobotAccuracy(np.asarray(robot_path), np.asarray(A_star_path))
            LoggedPixelGrid = pixel_grid
        else:
            Episode_Payload = [episode_count, last_episode_reward, last_episode_length, 1 if last_maze_solved else 0, sucess_rate, last_energy_used, last_regions_explored]
            ACCURACY = last_accuracy
            LoggedPixelGrid = last_pixel_grid

        action_payload = [sum_action_directions / network.MaxSteps, sum_action_steps_sizes / network.MaxSteps] 

        network.LogDataToCSV(Episode_Payload, network.actions.numpy(), action_payload, 'eps', agnt)
        network.LogDataToCSV(LoggedPixelGrid, ACCURACY, None, 'mze', agnt)

        episode_count += 1 
        episode_reward = 0 
        network.EnergyUsed = 0 
        episode_length = 0
        sum_action_directions = 0 
        sum_action_steps_sizes = 0 
            
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
                FinalState = GetState(robot_pos, walls)
                FinalState.append(float(np.linalg.norm(end_point - robot_pos)))
                FinalState.append(float(maze_size))
                FinalStateTensor = torch.FloatTensor(FinalState)
                _, _, next_value = network.forward(FinalStateTensor)
                next_value = next_value.squeeze(-1)

        # Calculate GAE
        advantages = torch.zeros(MaxSteps, dtype=torch.float32) 
        next_advantage = torch.tensor(0.0, dtype=torch.float32)

        for t in reversed(range(MaxSteps)): 
            next_value_t = next_value if t == MaxSteps - 1 else values[t + 1]
            done_flag = 1.0 - dones[t]
            temp_diff_error = rewards[t] + (Wgamma * next_value_t * done_flag) - values[t] 
            advantage = temp_diff_error + (Wgamma * Wlambda) * done_flag * next_advantage 
            advantages[t] = advantage 
            next_advantage = advantage

        advantages_st_dist, advantages_mean = torch.std_mean(advantages)
        returns = advantages + values
        advantages = (advantages - advantages_mean) / (advantages_st_dist + 1e-8) 

        NumberOfBeansInMyBowl = 4

        # Optimization loop
        for epoch in range(NumberOfBeansInMyBowl): 
            Shuffled_indices = torch.randperm(MaxSteps) 
            for i in range(0, MaxSteps, 64): 
                Batch = Shuffled_indices[i: i + 64] 
                Batch_states = states[Batch]
                Batch_actions = actions[Batch] 
                Batch_log_probs = log_probs[Batch] 

                b_returns = returns[Batch] 
                b_advantages = advantages[Batch] 

                b_new_log_prob, entropy, b_value = network.evaluate_actions(Batch_states, Batch_actions) 
                probability_ratio = torch.exp(b_new_log_prob - Batch_log_probs) 

                entropy = entropy.mean()

                surrogate_1 = probability_ratio * b_advantages 
                surrogate_2 = torch.clamp(probability_ratio, 0.8, 1.2) * b_advantages 

                ACTOR_LOSS = torch.mean(-torch.min(surrogate_1, surrogate_2)) 
                CRITIC_LOSS = F.mse_loss(b_value, b_returns) 
                TOTAL_LOSS = ACTOR_LOSS + network.BerryGoodKoreanDataHacker * CRITIC_LOSS - network.KoreanMumSpecial * entropy 

                optimizer.zero_grad() 
                TOTAL_LOSS.backward() 
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5) 
                optimizer.step() 

                # Dual update for turning penalty constraint multiplier
                avg_turn = total_turn_delta / MaxSteps
                w_turn_loss = -network.log_w_turn * (avg_turn - network.target_max_turn)
                w_optimizer.zero_grad()
                w_turn_loss.backward()
                w_optimizer.step()

                Training_Payload = [
                    steps,
                    sucess_rate,
                    TOTAL_LOSS.item(),
                    ACTOR_LOSS.item(),
                    CRITIC_LOSS.item(),
                    entropy.item(),
                    b_advantages.mean().item()
                ] 
                network.LogDataToCSV(Training_Payload, None, None, 'trg', agnt) 

        network.clear_memory(MaxSteps)

