# FOR THE FINAL TRAINING LOOP

from agent_ppo import StartAgent
import actor_critic
import agent_ppo
import PPO_action_converter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import json
from datetime import datetime, timedelta
import itertools
import os
import shutil
import time
import traceback
import torch

EPISODES = 100000
SEEDS = [42]
MACHINE = 1        
JOBS_BY_MACHINE = {1: range(0, 20), 2: range(20, 40), 3: range(40, 81)}   
MAX_WORKERS_BY_MACHINE = {1: 7, 2: 7, 3: 15}   
MAX_WORKERS = MAX_WORKERS_BY_MACHINE[MACHINE]
AGENTS = None
JOBS = [i for i in JOBS_BY_MACHINE[MACHINE] if AGENTS is None or i in AGENTS]
OUT = Path("runs") / "sweep_81_v5"
STOP_AT = None  
STOP_FILE = OUT / "STOP"   

DEFAULTS = dict(NNh1=48, NNh2=24, VARNumberOfRayCasts=24, VARAllowedEnergy=150, VARWeightInitType="orth",
                VARtargetMaxTurn=0.5, VARFOV=360, VARRayDistance=None)

GRID = {                                        
    "VARNumberOfRayCasts": [24, 16, 8],
    "VARFOV": [360, 180, 90],
    "VARAllowedEnergy": [150, 60, 30],
    "VARtargetMaxTurn": [3.14159, 1.0, 0.5],
}
CONSTANTS = {"MAX_STEP_START": 1.5, "MAX_STEP_END": 1.5}   

def stop_timestamp():
    if STOP_AT is None:
        return None
    hour, minute = map(int, STOP_AT.split(":"))
    now = datetime.now()
    stop = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if stop <= now:
        stop += timedelta(days=1)
    return stop.timestamp()

def build_jobs(stop_at=None):
    jobs = []
    for index, (levels, seed) in enumerate(itertools.product(itertools.product(*GRID.values()), SEEDS)):
        jobs.append(dict(index=index, seed=seed, episodes=EPISODES, params=dict(zip(GRID, levels)), stop_at=stop_at, stop_file=str(STOP_FILE),
                         directory=str(OUT / f"agent_{index:03d}_seed_{seed}")))
    return jobs

def set_constants(params):
    for name, value in params.items():
        if name in DEFAULTS:
            continue
        modules = [m for m in (actor_critic, agent_ppo, PPO_action_converter) if hasattr(m, name)]
        if not modules:
            raise KeyError(f"{name} is neither a StartAgent setting nor a constant in actor_critic, agent_ppo or PPO_action_converter")
        for m in modules:
            setattr(m, name, value)

def run_job(job):
    torch.set_num_threads(1)
    p = {**DEFAULTS, **job["params"]}
    directory = Path(job["directory"])
    summary_file = directory / f"Summary_sweep_{job['index']}.json"
    if summary_file.exists() and json.loads(summary_file.read_text(encoding="utf-8"))["episodes"] >= job["episodes"]:
        return job["index"]
    if (job["stop_at"] and time.time() >= job["stop_at"]) or Path(job["stop_file"]).exists():
        return job["index"]
    agent_ppo.STOP_AT, agent_ppo.STOP_FILE = job["stop_at"], job["stop_file"]
    try:
        set_constants({**CONSTANTS, **job["params"]})
        StartAgent("sweep_", p["NNh1"], p["NNh2"], p["VARNumberOfRayCasts"], p["VARAllowedEnergy"], p["VARNumberOfRayCasts"] + 6,
                   p["VARWeightInitType"], p["VARtargetMaxTurn"], job["index"], p["VARFOV"], p["VARRayDistance"],
                   seed=job["seed"], rollouts=job["episodes"], output_dir=directory)
    except Exception:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return job["index"]

start = time.perf_counter()
if __name__ == "__main__":
    all_jobs = build_jobs(stop_timestamp())
    jobs = [job for job in all_jobs if job["index"] in JOBS]
    OUT.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "seed", "directory"] + list(GRID))
        for job in all_jobs:
            writer.writerow([job["index"], job["seed"], job["directory"]] + list(job["params"].values()))
    with ProcessPoolExecutor(max_workers=min(len(jobs), MAX_WORKERS or os.cpu_count() or 1)) as pool:
        list(pool.map(run_job, jobs))
    failed = [job["index"] for job in jobs if (Path(job["directory"]) / "error.txt").exists()]
    if failed:
        (OUT / "failed.txt").write_text("\n".join(map(str, failed)), encoding="utf-8")
    print(f"finished: {time.perf_counter() - start}")


# Please go well!!!!!! - good night....