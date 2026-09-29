import numpy as np
import heapq
import time

class Tree:
    def choose_splitter(self, p0, p1, segment_vec, normal_vec):
            w_1 = 1  
            w_2 = 8
            heap = []

        # choose splitter loop
            for cand_idx, (cand_start, cand_end, cand_seg, cand_norm) in enumerate(zip(p0, p1, segment_vec, normal_vec)):
                
                front_count = 0
                back_count = 0
                split_count = 0
                
                for i, (start, end) in enumerate(zip(p0, p1)):
                    if i == cand_idx:
                        continue  
                    
                    #logic
                    start_score = np.dot(start - cand_seg, cand_norm)
                    end_score = np.dot(end - cand_seg, cand_norm)
        
                    if start_score > 1e-5 and end_score > 1e-5:
                       front_count += 1
                    elif start_score < -1e-5 and end_score < -1e-5:
                         back_count += 1
                    elif (start_score > -1e-5 and end_score < 1e-5) or (start_score < 1e-5 and end_score > -1e-5):
                         split_count += 1
                    else:
                        front_count += 1
        
                splitter_score = (w_1*abs(front_count - back_count) + (w_2*split_count))
                heapq.heappush(heap, (splitter_score, cand_idx))
                    
            priority, splitter_ = heapq.heappop(heap)
            
            return splitter_ #chosen splitter
            
    def create_bsp_tree(self, p0, p1, segment_vec, normal_vec, splitter_idx):

        tree_front_count = []
        tree_back_count = []
        tree_split_count = []
        
        splitter_mid_point = segment_vec[splitter_idx]
        splitter_normal = normal_vec[splitter_idx]

        for idx, (start, end, seg, norm) in enumerate(zip(p0, p1, segment_vec, normal_vec)):

            if idx == splitter_idx:
               tree_split_count.append([idx, start, end, seg, norm])
               continue

            target_point_start = start
            target_point_end = end

            classification_dot_end = np.dot(target_point_end - splitter_mid_point, splitter_normal)
            classification_dot_start = np.dot(target_point_start - splitter_mid_point, splitter_normal)

            if classification_dot_start > 1e-5 and classification_dot_end > 1e-5:
               tree_front_count.append([idx, start, end, seg, norm]) 
            elif classification_dot_start < -1e-5 and classification_dot_end < -1e-5:
                 tree_back_count.append([idx, start, end, seg, norm])
            elif (classification_dot_start > 1e-5 and classification_dot_end < -1e-5) or (classification_dot_start < -1e-5 and classification_dot_end > 1e-5):

                 intersection_parameter = classification_dot_start / (classification_dot_start - classification_dot_end)
                 intersection_point = target_point_start + intersection_parameter*(target_point_end - target_point_start)

                 subseg1 = target_point_start + 0.5 * (intersection_point - target_point_start)
                 subseg2 = intersection_point + 0.5 * (target_point_end - intersection_point)

                 if classification_dot_start > 0:
                
                    tree_front_count.append([idx, target_point_start, intersection_point, subseg1, norm])
                    tree_back_count.append([idx, intersection_point, target_point_end, subseg2, norm])

                 elif classification_dot_start < 0:
                     
                      tree_back_count.append([idx, target_point_start, intersection_point, subseg1, norm])
                      tree_front_count.append([idx, intersection_point, target_point_end, subseg2, norm])

            else:
                tree_split_count.append([idx, start, end, seg, norm])

        node = {
        'id':splitter_idx,
        'splitter_point': splitter_mid_point,      #node dictionary, helpful for tree recursion 
        'splitter_normal': splitter_normal,
        'coplanar': tree_split_count,
        'front': None,                      #stores sub-trees of front/back children
        'back': None }
    
        if tree_front_count:
           array_front = np.asarray(tree_front_count, dtype=object)
           
           front_p0 = array_front[:,1]
           front_p1 = array_front[:,2]
           front_seg = array_front[:,3]
           front_norm = array_front[:,4]

           front_splitter = self.choose_splitter(front_p0, front_p1, front_seg, front_norm)
           
           node['front'] = self.create_bsp_tree(front_p0, front_p1, front_seg, front_norm, front_splitter)


        if tree_back_count:
            array_back = np.asarray(tree_back_count, dtype=object)
            
            back_p0 = array_back[:,1]
            back_p1 = array_back[:,2]
            back_seg = array_back[:,3]
            back_norm = array_back[:,4]

            back_splitter = self.choose_splitter(back_p0, back_p1, back_seg, back_norm)

            node['back'] = self.create_bsp_tree(back_p0, back_p1, back_seg, back_norm, back_splitter)
            
        return node 

    def find_distance_to_node(self, node, robot_pos, facing_vec):
        splitter_norm = node['splitter_normal']
        splitter_point = node['splitter_point']
    
        pos_offset = robot_pos - splitter_point
        side_dot = np.dot(pos_offset, splitter_norm)

        if abs(side_dot) < 1e-5:
            position = "planar"
        elif side_dot > 0:
             position = "infront"
        else:
             position = "behind"

        best_d = None

        for segment in node['coplanar']:
            segment_start, segment_end = segment[1], segment[2]
            p = robot_pos
            r = facing_vec
            a = segment_start
            b = segment_end
            v = b - a
            
            denominator = r[0] * v[1] - r[1] * v[0]
            if abs(denominator) < 1e-5:
                continue 
                
            t = ((a[0] - p[0]) * v[1] - (a[1] - p[1]) * v[0]) / denominator
            s = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / denominator
            
            
            if t >= 0 and 0.0 <= s <= 1.0:
                if best_d is None or t < best_d:
                    best_d = t

        return np.array([best_d, position], dtype=object)
        
    def traverse_tree(self, current_node, robot_pos, facing_vec):

        if current_node is None:
           return float('inf')

        d, position = self.find_distance_to_node(current_node, robot_pos, facing_vec)

        if position == "infront":
           first_child = current_node['front']
           second_child = current_node['back']
        elif position == 'behind':
             first_child = current_node['back']
             second_child = current_node['front']
        else:
             distance_front = self.traverse_tree(current_node['front'], robot_pos, facing_vec)
             distance_back = self.traverse_tree(current_node['back'], robot_pos, facing_vec)
             return min(distance_front, distance_back)

        dist = self.traverse_tree(first_child, robot_pos, facing_vec)

        if dist is not None and dist < float('inf'):
           return dist

        if d is not None:
           return d

        return self.traverse_tree(second_child, robot_pos, facing_vec)
        

       
        
         
    
        

            