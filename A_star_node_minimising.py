import heapq
import numpy as np

def Turn_Penalized_A_star_search(grid: np.ndarray, source: any, destination: any, turn_penalty: float = 5.0) -> np.ndarray:
    source = np.array([int(source[0]), int(source[1])])
    destination = np.array([int(destination[0]), int(destination[1])])
    
    source_tuple = tuple(source.tolist())
    destination_tuple = tuple(destination.tolist())
    
    node_heap = []
    node_dictionary = {}
    node_parent_dictionary = {}
    visited_nodes = set()
    
    current_node = source
    current_node_tuple = source_tuple
    
    movement_costs = np.array([
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [1, -1], [-1, -1]
    ])

    g_costs = {source_tuple: 0.0}

    while not np.array_equal(current_node, destination):
        visited_nodes.add(current_node_tuple)
        current_g_cost = g_costs[current_node_tuple]

        parent_tuple = node_parent_dictionary.get(current_node_tuple, None)
        if parent_tuple is not None:
            incoming_dir = (current_node_tuple[0] - parent_tuple[0], current_node_tuple[1] - parent_tuple[1])
        else:
            incoming_dir = None

        for movement_cost in movement_costs:
            neibour_node = current_node + movement_cost

            if neibour_node[0] >= len(grid) or neibour_node[1] >= len(grid[0]) or neibour_node[0] < 0 or neibour_node[1] < 0:
                continue
            elif grid[neibour_node[0], neibour_node[1]] == 1:
                continue

            neibour_node_tuple = (int(neibour_node[0]), int(neibour_node[1]))

            if neibour_node_tuple in visited_nodes:
                continue

            step_dist = np.linalg.norm(movement_cost)
            
            outgoing_dir = (movement_cost[0], movement_cost[1])
            is_turn = (incoming_dir is not None) and (incoming_dir != outgoing_dir)
            
            penalty = turn_penalty if is_turn else 0.0

            g_cost = current_g_cost + step_dist + penalty
            h_cost = np.linalg.norm(destination - neibour_node)
            f_cost = g_cost + h_cost

            if neibour_node_tuple in node_dictionary:
                if node_dictionary[neibour_node_tuple] <= f_cost:
                    continue
                else:
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
    raw_path = [path_node]
    while path_node != source_tuple:
        path_node = node_parent_dictionary[path_node]
        raw_path.append(path_node)

    raw_path = raw_path[::-1]

    return np.array(raw_path)