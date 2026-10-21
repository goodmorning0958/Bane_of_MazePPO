# Run this from the project folder (the one with training_example_generator.py):  python dump_mazes.py
# It saves a few hundred mazes from the real maze generator to real_mazes.json, so they can be tested against the trained policies.
import json
import numpy as np
from training_example_generator import get_new_training_example

out = {}
for x in (51, 52, 53, 54):
    samples = []
    for _ in range(200):
        start, end, size, grid_distance, walls, grid = get_new_training_example(x)
        samples.append(dict(start=[float(start[0]), float(start[1])], end=[float(end[0]), float(end[1])], size=int(size),
                            grid_distance=int(grid_distance), grid=np.asarray(grid).astype(int).tolist()))
    out[str(x)] = samples

with open("real_mazes.json", "w") as f:
    json.dump(out, f)
print("saved real_mazes.json")