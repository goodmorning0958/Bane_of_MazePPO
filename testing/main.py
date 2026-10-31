# FOR THE FINAL TRAINING LOOP

from model.agent_ppo import StartAgent
from model import actor_critic
from model import agent_ppo
from environment import PPO_action_converter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import json
from datetime import datetime, timedelta
import itertools
import os
import time
import traceback
import torch

episodes = 100000
seeds = [42]
machine = 1        
jobs_by_machine = {1: range(0, 20), 2: range(20, 40), 3: range(40, 81)}   
max_workers_by_machine = {1: 7, 2: 7, 3: 15}   
max_workers = max_workers_by_machine[machine]

job_indices = [i for i in jobs_by_machine[machine]]
out = Path("training_results_v5")
stop_at = None  
stop_file = out / "STOP"   

defaults = dict(NNh1 = 48, NNh2 = 24, VARNumberOfRayCasts = 24, VARAllowedEnergy = 150, VARWeightInitType = "orth",
                VARtargetMaxTurn = 0.5, VARFOV = 360, VARRayDistance = None)

grid = {                                        
    "VARNumberOfRayCasts": [24, 16, 8],
    "VARFOV": [360, 180, 90],
    "VARAllowedEnergy": [150, 60, 30],
    "VARtargetMaxTurn": [3.14159, 1.0, 0.5],
}
constants = {"MAX_STEP_START": 1.5, "MAX_STEP_END": 1.5}   

def stop_timestamp():
    if stop_at is None:
        return None
    hour, minute = map(int, stop_at.split(":"))
    now = datetime.now()
    stop = now.replace(hour = hour, minute = minute, second = 0, microsecond = 0)
    if stop <= now:
        stop += timedelta(days = 1)
    return stop.timestamp()

def build_jobs(stop_at = None):
    jobs = []
    for index, (levels, seed) in enumerate(itertools.product(itertools.product(*grid.values()), seeds)):
        jobs.append(dict(index = index, seed = seed, episodes = episodes, params = dict(zip(grid, levels)), stop_at = stop_at, stop_file = str(stop_file),
                         directory = str(out / f"agent_{index:03d}_seed_{seed}")))
    return jobs

def set_constants(params):
    for name, value in params.items():
        if name in defaults:
            continue
        modules = [m for m in (actor_critic, agent_ppo, PPO_action_converter) if hasattr(m, name)]
        if not modules:
            raise KeyError(f"{name} is neither a StartAgent setting nor a constant in actor_critic, agent_ppo or PPO_action_converter")
        for m in modules:
            setattr(m, name, value)

def run_job(job):
    torch.set_num_threads(1)
    p = {**defaults, **job["params"]}

    directory = Path(job["directory"])
    summary_file = directory / f"Summary_sweep_{job['index']}.json"

    if summary_file.exists() and json.loads(summary_file.read_text(encoding = "utf-8"))["episodes"] >= job["episodes"]:
        return job["index"]
    
    if (job["stop_at"] and time.time() >= job["stop_at"]) or Path(job["stop_file"]).exists():
        return job["index"]
    
    agent_ppo.STOP_AT, agent_ppo.STOP_FILE = job["stop_at"], job["stop_file"]

    try:
        set_constants({**constants, **job["params"]})
        StartAgent("sweep_", p["NNh1"], p["NNh2"], p["VARNumberOfRayCasts"], p["VARAllowedEnergy"], p["VARNumberOfRayCasts"] + 6,
                   p["VARWeightInitType"], p["VARtargetMaxTurn"], job["index"], p["VARFOV"], p["VARRayDistance"],
                   seed = job["seed"], rollouts = job["episodes"], output_dir = directory)
    except Exception:
        directory.mkdir(parents = True, exist_ok = True)
        (directory / "error.txt").write_text(traceback.format_exc(), encoding = "utf-8")

    return job["index"]

start = time.perf_counter()

if __name__ == "__main__":
    all_jobs = build_jobs(stop_timestamp())
    jobs = [job for job in all_jobs if job["index"] in job_indices]

    out.mkdir(parents = True, exist_ok = True)
    stop_file.unlink(missing_ok = True)

    with open(out / "manifest.csv", "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "seed", "directory"] + list(grid))
        for job in all_jobs:
            writer.writerow([job["index"], job["seed"], job["directory"]] + list(job["params"].values()))
    with ProcessPoolExecutor(max_workers = min(len(jobs), max_workers or os.cpu_count() or 1)) as pool:
        list(pool.map(run_job, jobs))

    print(f"finished: {time.perf_counter() - start}")


# Please go well!!!!!! - good night....