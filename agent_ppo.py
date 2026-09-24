import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as dist 
import matplotlib.pyplot as plt 
import numpy as np 
import math 
import time 
import csv 

from Bane_of_mazePPO.training_example_generator import get_new_training_example 
from Bane_of_mazePPO.agent_viewer import Dashboard 
from Bane_of_mazePPO.raycaster import raycaster

raycaster_env = raycaster() 

class ActorCritic(nn.Module): 
    # input layer: 32 distances, distance_to_finish  
    # output layer: [direction, step_size] 
    def __init__(self, h1, h2, VARNumberOfRayCasts, VARAllowedEnergy, inputs, action_dim=2): 
        super().__init__() 
        self.MaxSteps = 200 
        self.clear_memory(self.MaxSteps) 
        self.EnergyUsed = 0 
        self.VARNumberOfRayCasts = VARNumberOfRayCasts 
        self.VARAllowedEnergy = VARAllowedEnergy 
        
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
        self.sd = nn.Parameter(torch.tensor(20)) 
        
        # Lagrangian constraints for Turn Energy Harshness
        self.log_w_turn = nn.Parameter(torch.tensor(math.log(0.05))) 
        self.target_max_turn = 0.1 

    @property
    def w_turn(self):
        # ensure w_turn remains positive
        return torch.exp(self.log_w_turn)

    @torch.no_grad() 
    def clear_memory(self, MaxSteps): 
        self.states = torch.zeros((MaxSteps, 34), dtype=torch.float32) 
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

    @torch.no_grad() 
    def get_action_and_value(self, state_list): # add exploration noise 
        state_tensor = torch.FloatTensor(state_list) 

        with torch.no_grad(): 
            mean, stds, value = self.forward(state_tensor) 
            dist_obj = dist.Normal(mean, stds) 
            action = dist_obj.sample() 
            log_prob = dist_obj.log_prob(action).sum(dim=-1) 
        
        return action.numpy(), log_prob, value, mean, stds 

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

        in_dead_end = (neibour_counter >= 3) 
        return in_dead_end, region_pos 
    
    @torch.no_grad() 
    def RunnIntoWall(self, walls, new_pos, robot_pos): 
        dx = new_pos[0] - robot_pos[0] 
        dy = new_pos[1] - robot_pos[1] 
        
        t_distance = raycaster_env.CastRay(robot_pos, walls, dx, dy) 
        
        return True if t_distance <= 1.0 else False 
        
    def reward(self, run_into_wall, in_dead_end, distance_finish, region_unexplored, robot_pos, new_pos, step_size, sucess_rate, robot_angle, new_angle): 
        Times = { 
            32: 0.0013145, 
            16: 0.0005158, 
            8: 0.0003281, 
            4: 0.0002455, 
        } 
        time = Times[self.VARNumberOfRayCasts] 
        # added a way for the robot to optimize its curriculum learning (sucess rate) 
        step_size = max(abs(step_size), 1e-5) 
        
        # Turn sharpness penalty calculation (heading difference wrapped to [-pi, pi])
        angle_diff = (new_angle - robot_angle + math.pi) % (2 * math.pi) - math.pi
        
        # USE .item() HERE SO PPO BACKPROP DOESNT INTERFERE WITH LAGRANGIAN UPDATE
        turn_penalty = self.w_turn.item() * abs(angle_diff) 
        
        reward = - (0.01 + time + turn_penalty) - ((1-sucess_rate) / 10) 
        self.EnergyUsed += time 

        if run_into_wall:  
            return reward - 10, True, False 

        if in_dead_end: 
            return reward - 10, False, False 
        
        if distance_finish < 0.5: # Needs a small threshold rather than strict == 0 
            return reward + 100, True, True 

        if self.EnergyUsed > self.VARAllowedEnergy: 
            reward -= 1 

        if region_unexplored: 
            reward += 0.1 

        if np.allclose(robot_pos, new_pos): 
            reward -= 10 

        return reward, False, False 
        
    def InitializeWeights(m, Type): # Types can be "Orth", "Kaim", or "Xaiv" 
        if isinstance(m, nn.Linear): 
            if Type == "orth": 
                nn.init.orthogonal_(m.weight, gain=torch.sqrt(2)) 
            elif Type == "Kaim": 
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu") 
            elif Type == "Xaiv": 
                nn.init.xavier_uniform_(m.weight) 
            else: 
                raise TypeError 

            nn.init(m.bias, 0.0) 

    @torch.no_grad() 
    def LogDataToCSV(Data1, Data2, Data3, CSVType): # Can be 'eps', or 'trg' 
        if CSVType == 'eps': 
            with open('EpisodeData.csv', mode='a', newline='') as file: 
                writer = csv.writer(file) 
                writer.writerow(Data1) 
            with open('RobotActions.csv', mode='a', newline='') as file: 
                writer = csv.writer(file) 
                writer.writerow(Data2) 
            with open('ActionData.csv', mode='a', newline='') as file: 
                writer = csv.writer(file) 
                writer.writerow(Data3) 
        else: 
            with open('TrainingData.csv', mode='a', newline='') as file: 
                writer = csv.writer(file) 
                writer.writerow(Data1) 

@torch.no_grad() 
def GetState(robot_pos, walls):  
    distances = [] 
    angle_step = (2 * math.pi) / 32 
    for dir_ in range(32): 
        distances.append(raycaster_env.CastRay(np.asarray(robot_pos), walls, math.cos(dir_ * angle_step), math.sin(dir_ * angle_step))) 
    return distances 


def StartAgent(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType): 
    if __name__ == '__main__': 
        past = 0 
        x = 0 
        not_past = 0 
        steps = 0 
        sucess_rate = 1 

        # stats varibles 
        episode_count = 0 
        episode_reward = 0 
        episode_length = 0 
        sum_action_directions = 0 
        sum_action_steps_sizes = 0 

        network = ActorCritic(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs) 
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-4) 
        
        # SEPARATE OPTIMIZER FOR LAGRANGIAN HARSHNESS MULTIPLIER
        w_optimizer = torch.optim.Adam([network.log_w_turn], lr=1e-3)
        
        dashboard = Dashboard() 
        dashboard.start() 

        # Initialize weights 
        network.InitializeWeights(VARWeightInitType) 

        # Main loop 
        while True: 
            start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd) 
            robot_pos = np.array(start_point, dtype=np.float32) 
            end_point = np.array(end_point, dtype=np.float32) 
            walls_np = np.array(walls) 
            maze_np = np.array(pixel_grid) 
            robot_angle = 0 
            
            total_turn_delta = 0.0

            network.clear_memory(network.MaxSteps)  
            maze_solved = False 
        
            RegionsExplored = set() 

            
            for step in range(network.MaxSteps):  
                # added for loop to stop an episode after a set amount of steps (to avoid the robot 
                # refusing to solve the maze) 
                
                # Get state (2d LiDAR) 
                State = GetState(robot_pos, walls)   
                State.append(np.linalg.norm(end_point - robot_pos)) 
                State.append(maze_size) 
                
                # forward pass 
                Action, log_prob, value, _, _, = network.get_action_and_value(State) 
                action_directions += Action[0] 
                action_steps_sizes += Action[1] 

                # virtually update the robot 
                angle, step_size = Action[0] % (2*math.pi), Action[1]  
                new_pos = robot_pos + step_size * np.array([np.cos(angle), np.sin(angle)]) 
                new_angle = angle 
                
                # Accummulate turn delta for lagrangian optimization
                angle_diff_step = abs((new_angle - robot_angle + math.pi) % (2 * math.pi) - math.pi)
                total_turn_delta += angle_diff_step

                # get state info 
                run_into_wall = network.RunnIntoWall(walls, new_pos, robot_pos) 
                in_dead_end, region_pos = network.InDeadEnd(grid_distance, new_pos, walls_np) 
            
                region_tuple = tuple(region_pos.tolist()) 
                region_unexplored = region_tuple not in RegionsExplored 
                if region_unexplored: 
                    RegionsExplored.add(region_tuple) 

                # Calculate agent reward 
                distance_finish = np.linalg.norm(end_point - new_pos) 
                Reward, done, maze_solved = network.reward(run_into_wall, in_dead_end, distance_finish, region_unexplored, robot_pos, new_pos, step_size, sucess_rate, robot_angle, new_angle) 
                episode_reward += Reward 
                # append to tuples 
                network.store_transition(State, Action, Reward, log_prob, done, value, step) 

                # update agent 
                dashboard.send(maze_np.tolist(), [[new_pos[0], new_pos[1], new_angle]]) 

                steps += 1 

                total_actions = max(1, past + not_past) 
                if total_actions == 0: 
                    sucess_rate = 1 
                else: 
                    sucess_rate = past / total_actions 
            
                if done: 
                    if maze_solved: 
                        past += 1 
                        x += 1 
                    else: 
                        not_past += 1 
                    
                    # Reset environment state without breaking rollout 
                    start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd) 
                    robot_pos = np.array(start_point, dtype=np.float32) 
                    end_point = np.array(end_point, dtype=np.float32) 
                    walls_np = np.array(walls) 
                    maze_np = np.array(pixel_grid) 
                    robot_angle = 0 
                    RE = RegionsExplored 
                    RegionsExplored = set() 
                else: 
                    robot_pos = new_pos 
                    robot_angle = new_angle 

                episode_length += 1 

            # a way for the AI to optimize its energy efficiency.
            avg_turn_delta = total_turn_delta / network.MaxSteps
            turn_constraint_loss = - network.w_turn * (avg_turn_delta - network.target_max_turn)
            
            w_optimizer.zero_grad()
            turn_constraint_loss.backward()
            w_optimizer.step()

            Episode_Payload = [episode_count, episode_reward, episode_length, 1 if maze_solved else 0, sucess_rate, network.EnergyUsed, len(RE)] 
            Actions = network.actions 
            action_payload = [sum_action_directions / episode_length, sum_action_steps_sizes / episode_length] 
            network.LogDataToCSV(Episode_Payload, Actions, action_payload, 'eps') 
   
            episode_count += 1 
            episode_reward = 0 
            network.EnergyUsed = 0 
            sum_action_directions = 0 
            sum_action_steps_sizes = 0 
            RE = set() 
                
                
            # innitialize the tensors here so network is never messed aroudn with for speed 
            states = network.states 
            actions = network.actions 
            rewards = network.rewards 
            log_probs = network.log_probs 
            dones = network.dones 
            values = network.values 
            MaxSteps = network.MaxSteps 

            Shuffled_indices = torch.randperm(MaxSteps) 
            Wgamma = 0.99 
            Wlambda = 0.99 

            next_advantege = 0 
            next_value = 0 
            next_done = 1 
                
            # Calculate GAE (advantages) 
            advantages = torch.zeros(MaxSteps, dtype=torch.float32) 
            for t in reversed(range(MaxSteps)): 
                # reversed GAE and temporial differincial error. Put the errors in the bag bro 
                done_flag = 1 - next_done 
                temp_diff_error = rewards[t] + (Wgamma * next_value * done_flag) - values[t] 
                advantage = temp_diff_error + (Wgamma * Wlambda) * done_flag * next_advantege 
                advantages[t] = advantage 
                
                next_advantege = advantage 
                next_value = values[t] 
                next_done = dones[t] 

            # normalize advantages 
            advantages_st_dist, advantages_mean = torch.std_mean(advantages) 
            advantages = (advantages - advantages_mean) / (advantages_st_dist + 1e-8) 
            CriticUpdationNorm = advantages + values 

            NumberOfBeansInMyBowl = 4 # number of epochs 

            # Training loop!!!! 
            for epoch in range(NumberOfBeansInMyBowl): 
                Shuffled_indices = torch.randperm(MaxSteps) 
                for i in range(0, MaxSteps, 64): # divide into mini batches for highly optimized progressive training 
                    Batch = Shuffled_indices[i: i + 64] 
                    Batch_states = states[Batch] 
                    Batch_log_probs = log_probs[Batch] 

                    # advanteges 
                    b_CriticUpdationNorm = CriticUpdationNorm[Batch] 
                    b_advantages = advantages[Batch] 

                    # clipped surrogate objective (actor loss, critic loss, total_loss) 
                    _, b_new_log_prob, b_value, mean, stds = network.get_action_and_value(Batch_states) 
                    probability_ratio = torch.exp(b_new_log_prob - Batch_log_probs) 

                    # components 
                    surrogate_1 = probability_ratio * b_advantages 
                    surrogate_2 = torch.clamp(probability_ratio, 0.8, 1.2) * b_advantages 

                    # entropy object: This connects the neural outputs to the total loss funciton 
                    dist_obj = torch.distributions.Normal(mean, stds) 
                    entropy = dist_obj.entropy().sum(dim=-1).mean() 

                    ACTOR_LOSS = torch.mean(-torch.min(surrogate_1, surrogate_2)) 
                    CRITIC_LOSS = F.mse_loss(b_value, b_CriticUpdationNorm) 
                    TOTAL_LOSS = ACTOR_LOSS + network.BerryGoodKoreanDataHacker * CRITIC_LOSS - network.KoreanMumSpecial * entropy 

                    # initialize your son and your daughter             
                    optimizer.zero_grad() 

                    # May the gradients be ever in your favor, as in never going backward, so go backward # backpropogation 
                    TOTAL_LOSS.backward() 

                    # gradient clipping 
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm=0.5) 

                    # Calculate Adam's Phone-bill Optimization 
                    optimizer.step() 

                    Training_Payload = [steps, sucess_rate, TOTAL_LOSS, ACTOR_LOSS, CRITIC_LOSS, entropy, b_advantages] 
                    network.LogDataToCSV(Training_Payload, None, None, 'trg') 

            network.clear_memory(MaxSteps)
