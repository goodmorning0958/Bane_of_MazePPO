from Policy_tester import StartAgent
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import itertools
import shutil
import time
import traceback
import torch

rollouts = 1800
max_workers = 7
agents = None
policies = Path("runs") / "sweep_81_v5"
out = Path("runs") / "eval_81_v5"

grid = {
    "VARNumberOfRayCasts": [24, 16, 8],
    "VARFOV": [360, 180, 90],
    "VARAllowedEnergy": [150, 60, 30],
    "VARtargetMaxTurn": [3.14159, 1.0, 0.5],
}

def run_test(job):
    torch.set_num_threads(1)
    index, p = job
    directory = out / f"agent_{index:03d}_eval_seed_42"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(policies / f"agent_{index:03d}_seed_42" / f"Policy_{index}.pt", directory)
    try:
        StartAgent("sweep_", 48, 24, p["VARNumberOfRayCasts"], p["VARAllowedEnergy"], p["VARNumberOfRayCasts"] + 6, "orth",
                   p["VARtargetMaxTurn"], index, p["VARFOV"], None, rollouts=rollouts, output_dir=directory)
    except Exception:
        (directory / "eval_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return index

if __name__ == "__main__":
    start = time.perf_counter()
    jobs = [(i, dict(zip(grid, levels))) for i, levels in enumerate(itertools.product(*grid.values()))]
    jobs = [j for j in jobs if (agents is None or j[0] in agents) and (policies / f"agent_{j[0]:03d}_seed_42" / f"Policy_{j[0]}.pt").exists()] 

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for done in as_completed([pool.submit(run_test, j) for j in jobs]):
            print(f"agent {done.result()} done ({(time.perf_counter() - start) / 60:.0f} min)")
    print(f"finished: {(time.perf_counter() - start) / 60:.0f} min")