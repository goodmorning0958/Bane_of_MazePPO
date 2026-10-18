import math
import numpy as np
from SImulated_BSP_LiDAR import Tree

# BSP tree
raycaster_env = Tree()


def _fov_to_radians(value):
    if value is None:
        return math.pi / 2
    value = float(value)
    if value > 2 * math.pi:
        value = math.radians(value)
    return min(max(value, 0.0), 2 * math.pi)

def _build_bsp_tree(walls):
    segments = np.asarray(walls, dtype=np.float32)

    if segments.size == 0:
        segments = segments.reshape(0, 2, 2)
    if segments.ndim == 2 and segments.shape[1] == 4:
        segments = segments.reshape(-1, 2, 2)

    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    valid = np.isfinite(segments).all(axis=(1, 2)) & (lengths > 1e-6)
    segments = segments[valid]
    
    p0 = segments[:, 0]
    p1 = segments[:, 1]

    dist_vec = p1 - p0

    segment_vec = p0 + 0.5 * dist_vec
    
    norm_val = np.linalg.norm(dist_vec, axis=1, keepdims=True)
    normal_vec = np.column_stack((-dist_vec[:, 1], dist_vec[:, 0])) / norm_val

    tree_splitter = raycaster_env.choose_splitter(p0, p1, segment_vec, normal_vec)
    bsp = raycaster_env.create_bsp_tree(p0, p1, segment_vec, normal_vec, tree_splitter)

    return segments, bsp