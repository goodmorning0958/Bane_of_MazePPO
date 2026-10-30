"""Environment diagnostics for the maze / LiDAR / collision / energy pipeline.

Run from the folder that holds your real files:
    training_example_generator.py, SImulated_BSP_LiDAR.py, maze_utils.py, RunIntoWall.py,
    PPO_action_converter.py, actor_critic.py
    python env_diagnostics.py

It only READS your modules (nothing is modified). Takes about a minute.
"""
import math
import collections
import numpy as np

from environment.training_example_generator import get_new_training_example
from environment.maze_utils import _build_bsp_tree, raycaster_env
from environment.RunIntoWall import RunIntoWall
from environment.PPO_action_converter import ConvertRobotAction

N_MAZES = 120          # mazes drawn for the generator / start-point / dead-end checks
N_BSP_MAZES = 30       # mazes used for the (slower) BSP check
X0 = 51                # training_steps value used for the maze draws in sections C-E (a real-maze stage)
AGENT_START_X = 0      # the x that agent_ppo.py starts with (line 130): 0 includes the empty-arena stage, 51 skips it
STAGES = (0, 25, 50, 51, 54, 148, 403, 1096)   # curriculum stages shown in section B
rng = np.random.default_rng(0)
np.random.seed(0)
ANGLES = np.linspace(-math.pi, math.pi, 24, endpoint=False)


def title(s):
    print("\n" + "=" * 8, s)


# ---------------------------------------------------------------- A. converter / energy
title("A. ConvertRobotAction: energy returned")
energies = {ConvertRobotAction([a, s], False)[1] for a in np.linspace(-math.pi, math.pi, 25) for s in (0.0, 1.0, 5.0)}
wheels = {tuple(round(v, 3) for v in ConvertRobotAction([a, 1.0], False)[0][0][:2]) for a in np.linspace(-math.pi, math.pi, 25)}
print("distinct energy values over 25 angles x 3 step sizes:", sorted(energies))
print("distinct (left, right) wheel commands over 25 angles:", sorted(wheels))
print("-> a single value means energy never changes with the action; a single wheel command means the steering branches are unreachable")

# ---------------------------------------------------------------- B. generator
title("B. get_new_training_example per curriculum stage")
draws = [get_new_training_example(X0, 5.89, 20.0) for _ in range(N_MAZES)]   # mazes used by sections C-E
N_STAGE = 40
print(f"{N_STAGE} draws per stage; x = training_steps value")
print(f"{'x':>5} {'kind':<12} {'maze_size':<10} {'grid_distance share':<28} {'grid!=size':>10} {'on wall':>8} {'start==goal':>12} {'1px corridors':>14}")
for x0 in STAGES:
    ds = [get_new_training_example(x0, 5.89, 20.0) for _ in range(N_STAGE)]
    kind = "empty arena" if x0 <= 50 else "maze"
    sizes = sorted({d[2] for d in ds})
    gd_share = collections.Counter(d[3] for d in ds)
    gd_txt = " ".join(f"{g}:{100 * v / N_STAGE:.0f}%" for g, v in sorted(gd_share.items()))
    size_mismatch = sum(d[5].shape[0] != d[2] for d in ds)
    on_wall = sum(d[5][int(d[0][0]), int(d[0][1])] != 0 or d[5][int(d[1][0]), int(d[1][1])] != 0 for d in ds)
    same = sum(bool(np.allclose(np.asarray(d[0], dtype=float), np.asarray(d[1], dtype=float))) for d in ds)
    one_px = "n/a" if x0 <= 50 else f"{100 * gd_share.get(1, 0) / N_STAGE:.0f}%"
    print(f"{x0:>5} {kind:<12} {str(sizes):<10} {gd_txt:<28} {size_mismatch:>10} {on_wall:>8} {same:>12} {one_px:>14}")
print("grid!=size: draws whose pixel grid is not maze_size x maze_size (LiDAR range / reward scale would be wrong); on wall: start or goal inside a wall pixel")
print("pixel grid sizes seen in the maze draws used below:", sorted({d[5].shape[0] for d in draws}))
print(f"agent_ppo.py starts at x = {AGENT_START_X}:",
      "the empty-arena stage (x <= 50) is reached" if AGENT_START_X <= 50 else "the empty-arena stage (x <= 50) is NEVER reached")
print("maze_size schedule (k=5.89):", {x: int(np.arange(10, 100)[np.argmax(np.exp(-0.5 * (((np.arange(10, 100) - 5.89 * math.floor(math.log(x + 1))) / 20.0) ** 2)))]) for x in (0, 51, 54, 148, 403, 1096)})

# ---------------------------------------------------------------- C. start / collision
title("C. Start point vs wall segments (RunIntoWall)")


def on_segment_endpoint(pos, walls):
    for p, q in walls:
        for e in (p, q):
            if np.allclose(pos, e, atol=1e-6):
                return True
    return False


def trapped_fraction(pos, walls):
    """Share of 24 directions in which a 0.01-long move already counts as a collision."""
    return float(np.mean([bool(RunIntoWall(pos, walls, 0.01 * math.cos(a), 0.01 * math.sin(a))) for a in ANGLES]))


res = collections.defaultdict(list)
for s, e, size, gd, walls, grid in draws:
    for name, off in (("as returned (pixel corner)", 0.0), ("+0.5 (pixel centre)", 0.5)):
        pos = np.array(s, dtype=np.float32) + off
        res[name].append((on_segment_endpoint(pos.astype(float), walls), trapped_fraction(pos, walls)))
for name, rows in res.items():
    print(f"{name:<28} start sits on a wall-segment end point: {100*np.mean([r[0] for r in rows]):5.1f}%   "
          f"mean share of directions blocked by a 0.01 move: {np.mean([r[1] for r in rows]):.2f}   "
          f"starts blocked in >=75% of directions: {100*np.mean([r[1] >= .75 for r in rows]):5.1f}%")

# ---------------------------------------------------------------- D. BSP vs brute force
title("D. BSP traverse_tree vs brute-force nearest hit (same segments, same ray test)")
rays = [np.array([math.cos(2 * math.pi * i / 24 + 0.1), math.sin(2 * math.pi * i / 24 + 0.1)], dtype=np.float32) for i in range(24)]


def brute(seg, p, r):
    best = float("inf")
    for a, b in seg:
        v = b - a
        den = r[0] * v[1] - r[1] * v[0]
        if abs(den) < 1e-5:
            continue
        t = ((a[0] - p[0]) * v[1] - (a[1] - p[1]) * v[0]) / den
        s = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den
        if t >= 0 and 0.0 <= s <= 1.0 and t < best:
            best = t
    return best


bad, tot, far, near = collections.Counter(), collections.Counter(), 0, 0
for s, e, size, gd, walls, grid in draws[:N_BSP_MAZES]:
    seg, bsp = _build_bsp_tree(walls)
    op = np.argwhere(grid == 0)
    for kind in ("start as returned", "pixel centre", "random interior"):
        for r_, c_ in op[rng.choice(len(op), 6, replace=False)]:
            if kind == "start as returned":
                p = np.array(s, dtype=np.float32)
            elif kind == "pixel centre":
                p = np.array([r_ + 0.5, c_ + 0.5], dtype=np.float32)
            else:
                p = np.array([r_ + rng.uniform(.05, .95), c_ + rng.uniform(.05, .95)], dtype=np.float32)
            for ray in rays:
                d1, d2 = raycaster_env.traverse_tree(bsp, p, ray), brute(seg, p, ray)
                tot[kind] += 1
                if not (d1 == d2 or (math.isfinite(d1) and math.isfinite(d2) and abs(d1 - d2) < 1e-4)):
                    bad[kind] += 1
                    far += d1 > d2
                    near += d1 < d2
for kind in tot:
    print(f"{kind:<20} wrong rays: {bad[kind]:>5} / {tot[kind]:<5} = {100*bad[kind]/tot[kind]:5.1f}%")
print(f"BSP distance too far: {far}   too near: {near}   (0 / 0 means the tree is exact)")

# ---------------------------------------------------------------- E. dead-end detector
title("E. InDeadEnd(grid_distance, pos, wall segments) vs true dead-end blocks")
try:
    from model.actor_critic import ActorCritic
    in_dead_end = lambda gd, pos, segs: ActorCritic.InDeadEnd(None, gd, pos, segs)
except Exception as exc:
    raise SystemExit(f"could not import ActorCritic ({exc}); skip section E")
conf = collections.Counter()
start_flagged = goal_flagged = 0
for s, e, size, gd, walls, grid in draws:
    b = gd if gd > 0 else 1
    base = np.asarray(grid)[::b, ::b]
    W = np.asarray(_build_bsp_tree(walls)[0], dtype=np.float32)
    for R in range(1, base.shape[0] - 1, 2):
        for C in range(1, base.shape[1] - 1, 2):
            if base[R, C] != 0:
                continue
            truth = (base[R - 1, C] + base[R + 1, C] + base[R, C - 1] + base[R, C + 1]) == 3
            pos = np.array([R * b + b / 2, C * b + b / 2], dtype=np.float32)
            conf[(bool(truth), bool(in_dead_end(gd, pos, W)[0]))] += 1
    start_flagged += bool(in_dead_end(gd, np.array(s, dtype=np.float32) + 0.5, W)[0])
    goal_flagged += bool(in_dead_end(gd, np.array(e, dtype=np.float32) + 0.5, W)[0])
tp, fn, fp = conf[(True, True)], conf[(True, False)], conf[(False, True)]
print(f"true dead ends: {tp+fn}   detected: {tp}   (recall {tp/max(tp+fn,1):.2f});  false alarms among {conf[(False,True)]+conf[(False,False)]} other cells: {fp}")
print(f"start block is a dead end in {100*start_flagged/len(draws):.0f}% of mazes, goal block in {100*goal_flagged/len(draws):.0f}%  -> agent_ppo must not treat the START block as a failure")

# ---------------------------------------------------------------- F. wall segments vs the pixel grid
title("F. RunIntoWall vs the pixel grid (does a step that crosses a wall pixel count as a collision?)")
bad = tot = 0
nseg = []
for s, e, size, gd, walls, grid in draws[:30]:
    nseg.append(len(walls))
    H, Wd = grid.shape
    op = np.argwhere(grid == 0)
    for r_, c_ in op[rng.choice(len(op), 25)]:
        p = np.array([r_ + .5, c_ + .5])
        for _ in range(8):
            a = rng.uniform(0, 2 * math.pi)
            L = rng.uniform(.2, 5.0)
            d = np.array([math.cos(a), math.sin(a)]) * L
            pts = p[None] + np.linspace(0, 1, int(L / 0.01) + 2)[:, None] * d[None]
            outside = any(q[0] < 0 or q[1] < 0 or q[0] >= H or q[1] >= Wd for q in pts)
            hit = any(q[0] < 0 or q[1] < 0 or q[0] >= H or q[1] >= Wd or grid[int(math.floor(q[0])), int(math.floor(q[1]))] == 1 for q in pts)
            got = bool(RunIntoWall(p, walls, d[0], d[1]))
            if hit and not got and outside:
                continue          # leaving the grid through a border opening is not a wall face
            tot += 1
            bad += hit != got
print(f"mismatches with the pixel truth: {bad} / {tot} steps;  mean wall segments per maze: {np.mean(nseg):.0f}")

print("\ndone")