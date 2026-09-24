from mazelib import Maze
from mazelib.generate.Prims import Prims
import random
import math
import heapq
import numpy as np
import torch
import torch.nn.functional as F

def generate_maze(grid_distance, maze_size):
    target_unscaled_size = maze_size // grid_distance
    
    h_w = max(3, (target_unscaled_size - 1) // 2)
    
    m = Maze()
    m.generator = Prims(h_w, h_w)
    m.generate()
    
    try:
        m.generate_entrances()
    except Exception:
        pass
        
    grid = m.grid
    start = tuple(m.start) if m.start is not None and len(m.start) == 2 else (1, 1)
    end = tuple(m.end) if m.end is not None and len(m.end) == 2 else (grid.shape[0] - 2, grid.shape[1] - 2)

    if grid_distance > 1:
        grid = np.repeat(np.repeat(grid, grid_distance, axis=0), grid_distance, axis=1)
        start = (start[0] * grid_distance, start[1] * grid_distance)
        end = (end[0] * grid_distance, end[1] * grid_distance)
    
    if grid.shape[0] < maze_size:
        pad_h = maze_size - grid.shape[0]
        pad_w = maze_size - grid.shape[1]
        grid = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1)
        
    return grid, start, end


def get_new_training_example(training_steps, k=5.89, sd=20):
    # Extract scalar float values if PyTorch Parameters are passed
    if isinstance(k, torch.Tensor):
        k = k.item()
    if isinstance(sd, torch.Tensor):
        sd = sd.item()

    # Cirruculum maze size generator (based on sucess steps of the robot)
    x = np.arange(10, 100) 
    size = k * np.floor(np.log(training_steps + 1))
    probabilities = np.exp(-0.5 * (((x - size) / sd) ** 2))
    maze_size = int(x[np.argmax(probabilities)])
    
    # generate grid_distance (again based on a prob distribution)
    grid_distances = [i for i in range(1, maze_size // 3 + 1) if maze_size % i == 0]
    
    if not grid_distances:
        grid_distance = 1  
    else:
        n = len(grid_distances)
        indices = np.arange(n)
        
        if maze_size >= 100:
            scale = n * 0.8 
            weights = np.exp(-indices / scale)
            
        elif maze_size <= 50:
            mid = n / 2.0
            sigma = max(n / 3.0, 1.0) 
            weights = np.exp(-((indices - mid) ** 2) / (2 * sigma ** 2))
            
        else:  
            weights = np.ones(n) 
        
        probabilities = weights / np.sum(weights)
        chosen_distance = np.random.choice(grid_distances, p=probabilities)
        grid_distance = int(chosen_distance)

    # We can start by giving the robot empty mazes to adapt/learn straight-line best path approaches
    if training_steps <= 50:
       grid_x, grid_y = torch.meshgrid(torch.arange(maze_size), torch.arange(maze_size), indexing='ij')

       Dist1 = (maze_size - 1) - grid_x
       Dist2 = (maze_size - 1) - grid_y
       DistEdge = torch.minimum(Dist1, Dist2)
            
       end_point_distribution = torch.exp(-(DistEdge.float() ** 2) / (2 * sd ** 2))
       end_point_distribution = F.softmax(end_point_distribution.view(-1), dim=-1).view(maze_size, maze_size)

       value = torch.multinomial(end_point_distribution.view(-1), num_samples=1).item()

       start_point = (0.0, 0.0)
       end_point = (float(value % maze_size), float(value // maze_size))
       walls = []
       pixel_grid = np.zeros((maze_size, maze_size), dtype=np.int32)

       return start_point, end_point, maze_size, grid_distance, walls, pixel_grid
    
    M = generate_maze(grid_distance, maze_size)

    pixel_grid = M[0]
    start_point = (float(M[1][0]), float(M[1][1]))
    end_point = (float(M[2][0]), float(M[2][1]))

    walls = []

    # Convert the maze to a list of line-segments
    for i in range(pixel_grid.shape[0]):
        queue = []
        for j in range(pixel_grid.shape[1]):
            if pixel_grid[i, j] == 1:
                queue.append(j)
            else:
                if queue:
                    walls.append([[i, queue[0]], [i, queue[-1] + 1]])
                    queue = []
        if queue:
            walls.append([[i, queue[0]], [i, queue[-1] + 1]])

    for j in range(pixel_grid.shape[1]):
        queue = []
        for i in range(pixel_grid.shape[0]):
            if pixel_grid[i, j] == 1:
                queue.append(i)
            else:
                if queue:
                    walls.append([[queue[0], j], [queue[-1] + 1, j]])
                    queue = []
        if queue:
            walls.append([[queue[0], j], [queue[-1] + 1, j]])

    return start_point, end_point, maze_size, grid_distance, walls, pixel_grid




        
         
    
        

            