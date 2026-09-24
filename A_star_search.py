import numpy as np
import heapq 

def Normal_A_star_search(grid: np.array, source: any, destination: any) -> np.array:
    source = np.array([int(source[0]), int(source[1])])
    destination = np.array([int(destination[0]), int(destination[1])])
    
    source_tuple = tuple(source.tolist())
    destination_tuple = tuple(destination.tolist())
    
    A_star_path = []
    
    node_heap = []
    node_dictionary = {}
    node_parent_dictionary = {}
    visited_nodes = set()
    
    current_node = source
    current_node_tuple = source_tuple
    movement_costs = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]])

    g_costs = {source_tuple: 0.0}

    while not np.array_equal(current_node, destination):
        visited_nodes.add(current_node_tuple)
        
        current_g_cost = g_costs[current_node_tuple]

        for movement_cost in movement_costs:
            neibour_node = current_node + movement_cost

            if neibour_node[0] >= len(grid) or neibour_node[1] >= len(grid[0]) or neibour_node[0] < 0 or neibour_node[1] < 0:
                continue
            elif grid[neibour_node[0], neibour_node[1]] == 1:
                continue

            neibour_node_tuple = (int(neibour_node[0]), int(neibour_node[1]))

            if neibour_node_tuple in visited_nodes:
                continue

            h_cost = np.linalg.norm(destination - neibour_node)
            g_cost = current_g_cost + np.linalg.norm(movement_cost)
            f_cost = g_cost + h_cost

            if neibour_node_tuple in node_dictionary:
                if node_dictionary[neibour_node_tuple] <= f_cost:
                    continue
                elif node_dictionary[neibour_node_tuple] > f_cost:
                    heapq.heappush(node_heap, (f_cost, neibour_node_tuple))
                    g_costs[neibour_node_tuple] = g_cost
                    node_dictionary[neibour_node_tuple] = f_cost
                    node_parent_dictionary[neibour_node_tuple] = current_node_tuple
            else:
                heapq.heappush(node_heap, (f_cost, neibour_node_tuple))
                g_costs[neibour_node_tuple] = g_cost
                node_dictionary[neibour_node_tuple] = f_cost
                node_parent_dictionary[neibour_node_tuple] = current_node_tuple

        found_valid_node = False
        while node_heap:
            best_node_cost, best_node_tuple = heapq.heappop(node_heap)
            if best_node_tuple not in visited_nodes:
                found_valid_node = True
                break

        if not found_valid_node:
            return np.array([])

        current_node_tuple = best_node_tuple
        current_node = np.array(best_node_tuple)

    path_node = destination_tuple
    A_star_path.append(list(path_node))
    while path_node != source_tuple:
        path_node = node_parent_dictionary[path_node]
        A_star_path.append(list(path_node))
    
    return np.flip(np.array(A_star_path), axis=0)