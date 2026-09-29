import heapq
import numpy as np
import random
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time

from training_example_generator import get_new_training_example
from A_star_search import Normal_A_star_search
from A_star_node_minimising import Turn_Penalized_A_star_search

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, 1010)
ax.set_ylim(0, 10)

(line,) = ax.plot([], [], "b-", marker="o", markerfacecolor="red")

x_data = []
y_data = []

for i in range(1000):
    start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(10000*i)
    
    start_row = min(max(0, int(start_point[0])), pixel_grid.shape[0] - 1)
    start_col = min(max(0, int(start_point[1])), pixel_grid.shape[1] - 1)
    
    end_row = min(max(0, int(end_point[0])), pixel_grid.shape[0] - 1)
    end_col = min(max(0, int(end_point[1])), pixel_grid.shape[1] - 1)

    pixel_grid[start_row, start_col] = 0
    pixel_grid[end_row, end_col] = 0

    Normal_path = Normal_A_star_search(np.array(pixel_grid), (start_row, start_col), (end_row, end_col))
    Energy_efficient_path = Turn_Penalized_A_star_search(np.array(pixel_grid), (start_row, start_col), (end_row, end_col))

    dtw_distance, alignment_path = fastdtw(Normal_path, Energy_efficient_path, dist=euclidean)
    Error = dtw_distance / len(alignment_path)

    print(Error)
    x_data.append(i)
    y_data.append(Error)

    line.set_data(x_data, y_data)
    if i >= ax.get_xlim()[1]:
        ax.set_xlim(0, i + 10)

    fig.canvas.draw()
    fig.canvas.flush_events()

Avg_error = np.sqrt(np.mean(y_data) ** 2)

plt.ioff()
plt.show()

print(Avg_error)