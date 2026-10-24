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

    start_mean = tuple(m.start) if m.start is not None and len(m.start) == 2 else (1, 1)
    end_mean = tuple(m.end) if m.end is not None and len(m.end) == 2 else (grid.shape[0] - 2, grid.shape[1] - 2)
    
    if grid_distance > 1:
        grid = np.repeat(np.repeat(grid, grid_distance, axis=0), grid_distance, axis=1)
        start_mean = (start_mean[0] * grid_distance, start_mean[1] * grid_distance)
        end_mean = (end_mean[0] * grid_distance, end_mean[1] * grid_distance)
    
    if grid.shape[0] < maze_size:
        pad_h = maze_size - grid.shape[0]
        pad_w = maze_size - grid.shape[1]
        grid = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1)

    x = np.arange(grid.shape[0])
    y = np.arange(grid.shape[0])
    X, Y = np.meshgrid(x, y, indexing='ij')

    # pick start and end from a probability distribution
    Start_Probabilities = np.exp(-((X - start_mean[0])**2 + (Y - start_mean[1])**2) / (2 * (maze_size / 20)**2))
    End_Probabilities = np.exp(-((X - end_mean[0])**2 + (Y - end_mean[1])**2) / (2 * (maze_size / 20)**2))

    # Make sure start and end don't fall on a wall
    Start_Probabilities *= 1 - np.array(grid)
    End_Probabilities *= 1 - np.array(grid)
    
    # normalize to make all probabilities add to 1
    Start_Probabilities /= np.sum(Start_Probabilities)
    End_Probabilities /= np.sum(End_Probabilities)

    Start_Probabilities = Start_Probabilities.flatten()
    End_Probabilities = End_Probabilities.flatten()

    # sample a start and end point
    start_indices = np.arange(Start_Probabilities.size)
    end_indices = np.arange(End_Probabilities.size)
    start = np.unravel_index(np.random.choice(start_indices, p=Start_Probabilities), grid.shape)
    end = np.unravel_index(np.random.choice(end_indices, p=End_Probabilities), grid.shape)
    
    return grid, list(start), list(end)

def add_walls(grid, start, end, n_walls, max_tries=200):
    size = grid.shape[0]
    start = np.array(start)
    end = np.array(end)
    for _ in range(max_tries):
        trial = grid.copy()
        placed = 0
        for _ in range(n_walls):
            length = np.random.randint(4, 4 + 2 * n_walls + 1)

            centre = np.round(start + np.random.uniform(0.3, 0.7) * (end - start)).astype(int) + np.random.randint(-2, 3, 2)
            half = np.arange(length) - length // 2
            if np.random.rand() < 0.5:
                cells = [(centre[0], centre[1] + h) for h in half]
            else:
                cells = [(centre[0] + h, centre[1]) for h in half]

            if all(3 <= r <= size - 4 and 3 <= c <= size - 4 and max(abs(r - start[0]), abs(c - start[1])) > 2 and max(abs(r - end[0]), abs(c - end[1])) > 2 for r, c in cells):
                for r, c in cells:
                    trial[r, c] = 1
                placed += 1
        if placed == n_walls and reachable(trial, start, end):
            return trial
    return grid

def reachable(grid, start, end):
    seen = {tuple(start)}
    queue = [tuple(start)]
    while queue:
        r, c = queue.pop()
        if (r, c) == tuple(end):
            return True
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return False

def get_new_training_example(training_steps, k=5.89, sd=20):
    if isinstance(k, torch.Tensor):
        k = k.item()
    if isinstance(sd, torch.Tensor):
        sd = sd.item()

    x = np.arange(10, 100) 
    size = k * np.floor(np.log(training_steps + 1))
    probabilities = np.exp(-0.5 * (((x - size) / sd) ** 2))
    maze_size = int(x[np.argmax(probabilities)])
    
    grid_distances = list(range(1, max(1, maze_size // 7) + 1))

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
        if training_steps in (51, 52):
            grid_distance = 3 if training_steps == 51 else 2

    if training_steps <= 50:
        x = np.arange(maze_size)
        y = np.arange(maze_size)
        X, Y = np.meshgrid(x, y)

        grid = np.zeros((maze_size, maze_size))
        grid[0] = 1
        grid[maze_size - 1] = 1
        grid[:, 0] = 1
        grid[:, maze_size - 1] = 1

        free = np.argwhere(grid == 0)
        n_bars = max(0, (training_steps - 20) // 10)
        for _ in range(1000):
            start = free[np.random.randint(len(free))]
            end = free[np.random.randint(len(free))]
            if np.linalg.norm(start - end) < maze_size / 2:
                continue
            if n_bars == 0:
                break
            trial = add_walls(grid, start, end, n_bars)       
            if trial.sum() > grid.sum():
                grid = trial
                break

        start_point = (start[0], start[1])
        end_point = (end[0], end[1])

        if training_steps <= 20:
            walls = [[[1, 1], [maze_size - 1, 1]], [[maze_size - 1, 1], [maze_size - 1, maze_size - 1]], [[maze_size - 1, maze_size - 1], [1, maze_size - 1]], [[1, maze_size - 1], [1, 1]]]

            return start_point, end_point, maze_size, grid_distance, walls, grid


        M = (grid, [start[0], start[1]], [end[0], end[1]])
    else:
        M = generate_maze(grid_distance, maze_size)

    pixel_grid = M[0]
    start_point = (float(M[1][0]), float(M[1][1]))
    end_point = (float(M[2][0]), float(M[2][1]))

    walls = []

    rows, cols = pixel_grid.shape

    # cornvert the grid into a list of wall segments
    for i in range(rows + 1):
        queue = []
        for j in range(cols):
            if 0 < i < rows and pixel_grid[i - 1, j] != pixel_grid[i, j]:
                queue.append(j)
            else:
                if queue:
                    walls.append([[i, queue[0]], [i, queue[-1] + 1]])
                    queue = []
        if queue:
            walls.append([[i, queue[0]], [i, queue[-1] + 1]])

    for j in range(cols + 1):
        queue = []
        for i in range(rows):
            if 0 < j < cols and pixel_grid[i, j - 1] != pixel_grid[i, j]:
                queue.append(i)
            else:
                if queue:
                    walls.append([[queue[0], j], [queue[-1] + 1, j]])
                    queue = []
        if queue:
            walls.append([[queue[0], j], [queue[-1] + 1, j]])

    return start_point, end_point, pixel_grid.shape[0], grid_distance, walls, pixel_grid