import heapq
import numpy as np
import random

from agent_viewer import get_new_training_example
from A_star_search import Normal_A_star_search
from A_star_node_minimising import Turn_Penalized_A_star_search


for _ in range(100):
    start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(random.randint(100, 10000000))
    
    start_row = min(max(0, int(start_point[0])), pixel_grid.shape[0] - 1)
    start_col = min(max(0, int(start_point[1])), pixel_grid.shape[1] - 1)
    
    end_row = min(max(0, int(end_point[0])), pixel_grid.shape[0] - 1)
    end_col = min(max(0, int(end_point[1])), pixel_grid.shape[1] - 1)

    pixel_grid[start_row, start_col] = 0
    pixel_grid[end_row, end_col] = 0

    Normal_path = Normal_A_star_search(np.array(pixel_grid), (start_row, start_col), (end_row, end_col))
    Energy_efficient_path = Turn_Penalized_A_star_search(np.array(pixel_grid), (start_row, start_col), (end_row, end_col))
    
