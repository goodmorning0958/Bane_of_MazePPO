"""
agent_ppo_pygame.py - pygame viewer that REPLACES the main loop of agent_ppo.StartAgent.

What you see
    * the maze (walls light, floor dark), the start block (blue) and goal block (green)
    * the robot (orange arrow), its trail, and the last move (white; red if it hit a wall)
    * the LiDAR rays exactly as the network receives them:
        red -> amber -> cyan = short -> long, a dot = where the ray hit a wall, a dim cyan ray = no hit within
        VARRayDistance, a short grey tick = a ray outside the FOV that was not cast
    * a side panel with episode / step / reward / energy / success rate / PPO status

How it relates to agent_ppo.py
    The episode and step code below is the body of StartAgent copied over, with one `yield` after the
    LiDAR scan of every step so the window can draw it. PPO updates call agent_ppo._compute_gae and
    agent_ppo._ppo_update, so the constants in agent_ppo.py (MIN_ROLLOUT_STEPS, TARGET_KL, ...) apply here
    too. No CSV files are written and nothing in the other scripts is changed.

Usage
    from main.py (same arguments as StartAgent, minus nothing):
        from agent_ppo_pygame import StartViewer
        StartViewer("test", 48, 24, 24, 9000, 34, "orth", 0.5, 1, 360, None)
    or from a terminal in the project folder:
        python agent_ppo_pygame.py                              train and watch live
        python agent_ppo_pygame.py --start-x 51                 skip the empty-arena stage
        python agent_ppo_pygame.py --policy runs/<run>/Policy_1.pt     watch a saved policy (no training)

Keys
    SPACE pause      N / RIGHT one step      UP / DOWN speed      F turbo (no waiting, draws once per frame)
    S new maze       [ / ] curriculum stage down / up (new maze)
    L rays   T trail   G block grid   W wall segments   E explored blocks   H help   P screenshot   Q / Esc quit
"""
import functools
import inspect
import itertools
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pygame

from training_example_generator import get_new_training_example
from RunIntoWall import RunIntoWall
from actor_critic import ActorCritic, ADAPT_TURN_WEIGHT
from maze_utils import _build_bsp_tree
from agent_ppo import (
    _compute_gae,
    _ppo_update,
    SEED,
    MIN_ROLLOUT_STEPS,
    MIN_ROLLOUT_STEPS_LEARNING,
    LEARNING_RATE,
    CURRICULUM_ENABLED,
    TURN_LIMIT_SUCCESS,
    CURRICULUM_WINDOW,
    CURRICULUM_PROMOTE,
    CURRICULUM_DEMOTE,
    CURRICULUM_STAGES,
    STALL_WINDOW,
    STALL_MIN_DISPLACEMENT,
    STALL_END_PENALTY,
)

# ---------------------------------------------------------------- look & feel
BG, PANEL, FLOOR, WALL, GRID = (16, 18, 24), (24, 27, 36), (22, 26, 34), (92, 104, 130), (46, 54, 70)
TEXT, DIM = (214, 220, 232), (128, 138, 158)
ROBOT, TRAIL, GOAL, START = (255, 170, 40), (255, 214, 120), (70, 220, 120), (90, 150, 255)
RAY_NEAR, RAY_MID, RAY_FAR, RAY_OFF = (255, 80, 70), (255, 210, 80), (70, 210, 255), (64, 70, 86)
BAD, WARN, SEGMENT, REGION = (240, 70, 70), (255, 150, 40), (255, 200, 60), (30, 54, 50)
OUTCOME_COLOURS = {"success": GOAL, "collision": BAD, "dead_end": WARN, "stalled": (190, 120, 255), "time_limit": DIM}
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"

SPEEDS = [1, 2, 5, 10, 20, 50, 100, 200, 500]     # steps per second
HOLD_SECONDS = 0.6                                 # pause on the final frame of an episode
EMPTY_ARENA_UNTIL = 50                             # mirrors `training_steps <= 50` in training_example_generator

HELP = [
    "SPACE pause     N/RIGHT step",
    "UP/DOWN speed   F turbo",
    "S new maze      [ ] stage",
    "L rays  T trail  G blocks",
    "W walls E explored  P shot",
    "H help          Q/Esc quit",
]


# ---------------------------------------------------------------- geometry helpers
def ray_max_distance(maze_size, ray_distance):
    """Same rule as ActorCritic.GetState uses for the longest ray."""
    max_distance = float(maze_size if ray_distance is None else ray_distance)
    if not math.isfinite(max_distance) or max_distance <= 0:
        max_distance = max(float(maze_size), 1.0)
    return max_distance


def lidar_rays(state, ray_offset, angle, ray_count, fov, max_distance):
    """Rebuild the rays the network was given from its state vector.

    Returns [(world_angle, length, cast, hit)]. Uses the formulas of ActorCritic.GetState, so what is drawn
    is what the network saw (no second BSP traversal)."""
    step = (2 * math.pi) / ray_count
    rays = []
    for i in range(ray_count):
        ray_angle = step * i + ray_offset
        diff = abs((ray_angle - float(angle) + math.pi) % (2 * math.pi) - math.pi)
        cast = diff <= fov / 2
        length = float(state[i]) * max_distance
        rays.append((ray_angle, length, cast, bool(cast and length < max_distance - 1e-6)))
    return rays


def curriculum_stage_starts(k, sd, x_max=6000):
    """First curriculum counter of each stage: where the maze size changes, plus the first real maze."""
    sizes = np.arange(10, 100)
    starts, last = [0], None
    for x in range(x_max):
        size = k * np.floor(np.log(x + 1))
        maze = int(sizes[np.argmax(np.exp(-0.5 * (((sizes - size) / sd) ** 2)))])
        if last is not None and maze != last:
            starts.append(x)
        last = maze
    starts.append(EMPTY_ARENA_UNTIL + 1)
    return sorted(set(starts))


# ---------------------------------------------------------------- drawing
def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def ray_colour(fraction):
    """0 = touching the robot (red) ... 1 = full ray length (cyan)."""
    if fraction < 0.5:
        return _lerp(RAY_NEAR, RAY_MID, fraction * 2)
    return _lerp(RAY_MID, RAY_FAR, (fraction - 0.5) * 2)


def _heading(theta, length=1.0):
    """Screen direction of a heading. Positions are (row, col): +row is DOWN the screen, +col is RIGHT."""
    return math.sin(theta) * length, math.cos(theta) * length


def make_layout(screen_size, rows, cols, panel_w):
    width, height = screen_size
    avail_w, avail_h = width - panel_w - 30, height - 20
    cell = max(1, min(avail_w // cols, avail_h // rows))     # whole pixels per maze pixel: no seams
    return SimpleNamespace(
        cell=cell, rows=rows, cols=cols, panel_w=panel_w, panel_x=width - panel_w - 10,
        ox=10 + max(0, (avail_w - cell * cols) // 2), oy=10 + max(0, (avail_h - cell * rows) // 2),
    )


def to_screen(layout, p):
    """World (row, col) -> screen (x, y)."""
    return layout.ox + float(p[1]) * layout.cell, layout.oy + float(p[0]) * layout.cell


def _block_rect(layout, region, size):
    r0, c0 = region
    return (layout.ox + c0 * layout.cell, layout.oy + r0 * layout.cell, size * layout.cell, size * layout.cell)


def _draw_board(screen, layout, view, ui, cache):
    cell = layout.cell
    key = (view.episode_id, cell)
    if cache.get("key") != key:                                   # maze image is rebuilt once per episode
        surface = pygame.Surface((layout.cols * cell, layout.rows * cell))
        surface.fill(FLOOR)
        for i in range(layout.rows):
            for j in range(layout.cols):
                if view.grid[i, j] == 1:
                    surface.fill(WALL, (j * cell, i * cell, cell, cell))
        cache["key"], cache["surface"] = key, surface
    screen.blit(cache["surface"], (layout.ox, layout.oy))

    gd = view.grid_distance
    if ui.show_regions and gd > 0:
        for r0, c0 in view.explored:                              # blocks the robot has entered (open pixels only)
            r, c = int(r0), int(c0)
            for i in range(max(r, 0), min(r + gd, layout.rows)):
                for j in range(max(c, 0), min(c + gd, layout.cols)):
                    if view.grid[i, j] == 0:
                        screen.fill(REGION, (layout.ox + j * cell, layout.oy + i * cell, cell, cell))
    if ui.show_grid and gd * cell >= 6:
        for k in range(0, layout.rows + 1, gd):
            y = layout.oy + k * cell
            pygame.draw.line(screen, GRID, (layout.ox, y), (layout.ox + layout.cols * cell, y), 1)
        for k in range(0, layout.cols + 1, gd):
            x = layout.ox + k * cell
            pygame.draw.line(screen, GRID, (x, layout.oy), (x, layout.oy + layout.rows * cell), 1)
    if gd > 0 and view.start_region is not None:
        pygame.draw.rect(screen, START, _block_rect(layout, view.start_region, gd), 2)
        pygame.draw.rect(screen, GOAL, _block_rect(layout, view.goal_region, gd), 2)
    if view.dead_end and view.region is not None:
        pygame.draw.rect(screen, WARN, _block_rect(layout, view.region, gd), 3)

    if ui.show_segments:                                          # the segments the LiDAR actually sees
        for a, b in view.walls:
            pygame.draw.line(screen, SEGMENT, to_screen(layout, a), to_screen(layout, b), 1)

    if ui.show_trail and len(view.path) >= 2:
        pygame.draw.lines(screen, TRAIL, False, [to_screen(layout, p) for p in view.path], 2)
    if view.last_move is not None:
        pygame.draw.line(screen, BAD if view.collision else (255, 255, 255),
                         to_screen(layout, view.last_move[0]), to_screen(layout, view.last_move[1]), 2)

    pygame.draw.circle(screen, START, to_screen(layout, view.path[0]), max(3, cell // 4), 1)
    goal_xy = to_screen(layout, view.end_point)
    pygame.draw.circle(screen, GOAL, goal_xy, max(4, int(cell * 0.45)), 2)
    pygame.draw.circle(screen, GOAL, goal_xy, 2)

    origin = to_screen(layout, view.pos)
    if ui.show_rays and view.rays:
        for ray_angle, length, cast, hit in view.rays:
            if not cast:
                tick = _heading(ray_angle, 0.5 * cell)
                pygame.draw.line(screen, RAY_OFF, origin, (origin[0] + tick[0], origin[1] + tick[1]), 1)
                continue
            colour = ray_colour(length / view.max_ray) if hit else (40, 120, 150)
            end = (origin[0] + math.sin(ray_angle) * length * cell, origin[1] + math.cos(ray_angle) * length * cell)
            pygame.draw.line(screen, colour, origin, end, 1)
            if hit:
                pygame.draw.circle(screen, colour, end, 2)

    size = max(6, int(cell * 0.55))
    nose, left, right = _heading(view.angle, size), _heading(view.angle + 2.5, size * 0.7), _heading(view.angle - 2.5, size * 0.7)
    pygame.draw.polygon(screen, ROBOT, [(origin[0] + dx, origin[1] + dy) for dx, dy in (nose, left, right)])
    pygame.draw.circle(screen, (255, 255, 255), origin, 2)
    if view.collision:
        for dx, dy in ((1, 1), (1, -1)):
            pygame.draw.line(screen, BAD, (origin[0] - 6 * dx, origin[1] - 6 * dy), (origin[0] + 6 * dx, origin[1] + 6 * dy), 3)


def _draw_panel(screen, layout, fonts, view, ui):
    x, y = layout.panel_x, 12
    screen.fill(PANEL, (x - 8, 6, layout.panel_w, screen.get_height() - 12))

    def row(label, value, colour=TEXT):
        nonlocal y
        screen.blit(fonts.small.render(label, True, DIM), (x, y))
        screen.blit(fonts.small.render(str(value), True, colour), (x + 74, y))
        y += 18

    status = "FINISHED" if view.finished else ("PAUSED" if ui.paused else ("TURBO" if ui.turbo else f"x{SPEEDS[ui.speed]}/s"))
    row("mode", f"{view.mode}   {status}", WARN if ui.paused else TEXT)
    row("episode", f"#{view.episode}   step {view.step + 1}/{view.max_steps}")
    row("maze", f"{view.rows}x{view.cols}   block {view.grid_distance}")
    row("curric.", f"stage {view.stage}   x = {view.x}")
    row("rolling", f"{view.rolling_success:.2f}  (last {view.rolling_n})   ramp {view.ramp_rate:.2f}")
    row("success", f"{view.success_rate:.3f}  ({view.past}/{view.past + view.not_past})")
    mean_steps = sum(view.recent_steps) / len(view.recent_steps) if view.recent_steps else 0.0
    row("steps/ep", f"{mean_steps:.1f}  (last {len(view.recent_steps)})")
    y += 6
    row("robot", f"r {view.pos[0]:.2f}  c {view.pos[1]:.2f}")
    row("heading", f"{math.degrees(view.angle):+.1f} deg")
    row("to goal", f"{view.goal_distance:.2f} px")
    row("turn", f"{view.turn_deg:+.1f} deg   (limit {view.turn_limit_deg:.0f})")
    row("step", f"{view.step_size:.2f} px")
    y += 6
    row("reward", f"{view.reward:+.3f}   ep {view.episode_reward:+.2f}")
    row("energy", f"{view.energy_used:.1f} / {view.energy_budget:.0f}")
    row("value", f"{view.value:+.3f}")
    row("lidar", f"{view.rays_cast} rays")
    row("nearest", "-" if view.min_ray is None else f"{view.min_ray:.2f} px")
    row("wall hit", "YES" if view.collision else "no", BAD if view.collision else TEXT)
    row("dead end", "YES" if view.dead_end else "no", WARN if view.dead_end else TEXT)
    row("blocks", f"{len(view.explored)} explored")
    if view.train:
        row("ppo", f"buffer {view.buffered}/{view.rollout_target}   updates {view.updates}")
        row("last KL", "-" if view.last_kl is None else f"{view.last_kl:.4f}")
    y += 8
    screen.blit(fonts.small.render("recent episodes", True, DIM), (x, y))
    y += 20
    per_row = max(1, (layout.panel_w - 24) // 14)
    for i, outcome in enumerate(view.recent):
        screen.fill(OUTCOME_COLOURS.get(outcome, DIM), (x + (i % per_row) * 14, y + (i // per_row) * 14, 11, 11))
    y += 14 * (1 + (view.recent.maxlen - 1) // per_row) + 10
    legend = [("success", GOAL), ("wall", BAD), ("dead end", WARN), ("time", DIM)]
    for i, (name, colour) in enumerate(legend):
        screen.fill(colour, (x + (i % 2) * 120, y + (i // 2) * 16 + 3, 9, 9))
        screen.blit(fonts.small.render(name, True, DIM), (x + (i % 2) * 120 + 14, y + (i // 2) * 16))
    y += 40
    if ui.show_help:
        for line in HELP:
            screen.blit(fonts.small.render(line, True, DIM), (x, y))
            y += 17


def draw_frame(screen, fonts, view, ui, cache):
    screen.fill(BG)
    panel_w = 330 if screen.get_width() >= 800 else 250
    if view.grid is None:
        screen.blit(fonts.big.render("starting...", True, TEXT), (20, 20))
        return
    layout = make_layout(screen.get_size(), view.rows, view.cols, panel_w)
    ui.layout = layout
    _draw_board(screen, layout, view, ui, cache)
    _draw_panel(screen, layout, fonts, view, ui)
    if view.banner:
        text, colour = view.banner
        label = fonts.big.render(text, True, (10, 12, 16))
        w, h = label.get_size()
        bx, by = layout.ox + (layout.cols * layout.cell - w) // 2, layout.oy + 8
        screen.fill(colour, (bx - 10, by - 4, w + 20, h + 8))
        screen.blit(label, (bx, by))


def _new_view(train, mode, energy_budget):
    view = SimpleNamespace(
        # maze / episode
        grid=None, rows=0, cols=0, walls=[], grid_distance=1, maze_size=0, x=0, episode=0, episode_id=0,
        end_point=None, start_region=None, goal_region=None, explored=set(), path=[], max_steps=0, step=0,
        # robot
        pos=None, angle=0.0, rays=None, max_ray=1.0, last_move=None, collision=False, dead_end=False, region=None,
        turn_deg=0.0, turn_limit_deg=180.0, step_size=0.0, goal_distance=0.0,
        # numbers
        reward=0.0, episode_reward=0.0, energy_used=0.0, energy_budget=energy_budget, value=0.0,
        lidar_ms=0.0, rays_cast=0, min_ray=None, success_rate=0.0, past=0, not_past=0,
        buffered=0, updates=0, last_kl=None, rollout_target=MIN_ROLLOUT_STEPS,
        stage=0, rolling_success=0.0, rolling_n=0, ramp_rate=0.0,
        # bookkeeping
        banner=None, finished=False, train=train, mode=mode,
        recent=deque(maxlen=60), recent_steps=deque(maxlen=50),
    )
    view.update = view.__dict__.update          # view.update(a=1, b=2) sets several fields at once
    return view


# ---------------------------------------------------------------- the viewer (replaces StartAgent's loop)
def StartViewer(test, NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, NNinputs, VARWeightInitType, VARtargetMaxTurn, agnt, VARFOV, VARRayDistance,
                *, seed=SEED, rollouts=None, train=None, policy_path=None, deterministic=None, start_difficulty=0,
                steps_per_second=15, window=(1150, 700), max_frames=None, screenshot_dir="viewer_shots", on_frame=None):
    """Same positional arguments as StartAgent.

    rollouts          number of episodes to run; None (default) = until you close the window.
    train             True: PPO updates exactly like StartAgent. Default: True, or False when policy_path is given.
    policy_path       a Policy_*.pt written by StartAgent. Its saved config replaces the network arguments.
    deterministic     watch mode only: True = use the mean action (default), False = sample like training.
    start_difficulty  initial curriculum counter x (51 or more skips the empty arenas).
    max_frames / on_frame   auto-exit after N frames / a callback(screen, view, ui) for tests and recordings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint = None
    if policy_path is not None:
        checkpoint = torch.load(str(policy_path), map_location="cpu")
        saved = checkpoint.get("config", {})
        NNh1, NNh2, NNinputs = saved.get("h1", NNh1), saved.get("h2", NNh2), saved.get("inputs", NNinputs)
        VARNumberOfRayCasts = saved.get("rays", VARNumberOfRayCasts)
        VARAllowedEnergy = saved.get("energy_budget", VARAllowedEnergy)
        VARtargetMaxTurn = saved.get("target_max_turn", VARtargetMaxTurn)
        VARFOV = saved.get("fov_radians", VARFOV)
        VARRayDistance = saved.get("ray_distance", VARRayDistance)
    if train is None:
        train = checkpoint is None
    deterministic = (not train) if deterministic is None else (bool(deterministic) and not train)

    # classes (same as StartAgent)
    network = ActorCritic(NNh1, NNh2, VARNumberOfRayCasts, VARAllowedEnergy, VARtargetMaxTurn, VARFOV, NNinputs)
    network.InitializeWeights(VARWeightInitType)
    if checkpoint is not None:
        network.load_state_dict(checkpoint["model_state_dict"])
        if "max_step" in checkpoint:                        # the step cap the policy finished training with
            network.max_step = float(checkpoint["max_step"])
            network.step_scale = network.max_step / 2.0
    else:
        network.set_max_step(0.0)                           # same as StartAgent
    ppo_parameters = [p for name, p in network.named_parameters() if p.requires_grad and name != "log_w_turn"]
    optimizer = torch.optim.Adam(ppo_parameters, lr=LEARNING_RATE, eps=1e-5)
    w_optimizer = torch.optim.Adam([network.log_w_turn], lr=1e-3) if ADAPT_TURN_WEIGHT else None

    # ActorCritic.EnergyCost takes (VARRayDistance, maze_size). If reward() does not pass them on, the viewer supplies
    # them here, so actor_critic.py stays untouched. If reward() passes them, or EnergyCost no longer takes them,
    # this changes nothing.
    reward_params = inspect.signature(network.reward).parameters
    energy_cost = network.EnergyCost
    energy_cost_needs_ray_args = "VARRayDistance" in inspect.signature(energy_cost).parameters

    mode = "TRAIN" if train else ("WATCH (mean)" if deterministic else "WATCH (sampled)")
    view = _new_view(train, mode, float(VARAllowedEnergy))
    view.max_steps = network.MaxSteps
    ctl = SimpleNamespace(skip=False, set_x=None)
    stages = curriculum_stage_starts(network.k.item(), network.sd.item())

    def stage_of(x_value):
        return max(i for i, start in enumerate(CURRICULUM_STAGES) if start <= x_value)

    def episodes():
        x = int(start_difficulty)
        stage = stage_of(x)
        success_rate = 0.0
        episode_count = 0
        past = 0
        not_past = 0
        recent_outcomes = deque(maxlen=CURRICULUM_WINDOW)   # same bookkeeping as StartAgent
        stage_outcomes = deque(maxlen=CURRICULUM_WINDOW)
        ramp_rate = 0.0
        capacity = max(MIN_ROLLOUT_STEPS, MIN_ROLLOUT_STEPS_LEARNING) + network.MaxSteps
        network.clear_memory(capacity)
        buffered_steps = 0
        buffered_advantages, buffered_returns = [], []
        buffer_turn_total = 0.0
        updates = 0
        episode_ids = range(rollouts) if rollouts is not None else itertools.count()

        if energy_cost_needs_ray_args:                      # fill in whatever reward() does not pass (maze_size is read per call)
            def energy_cost_with_ray_args(*args, **kwargs):
                if len(args) < 9:
                    kwargs.setdefault("VARRayDistance", VARRayDistance)
                    kwargs.setdefault("maze_size", maze_size)
                return energy_cost(*args, **kwargs)
            network.EnergyCost = energy_cost_with_ray_args

        for _rollout in episode_ids:
            if ctl.set_x is not None:                       # [ / ] keys
                x, ctl.set_x = ctl.set_x, None
                stage = stage_of(x)
                stage_outcomes.clear()
            ctl.skip = False

            # get training example
            start_point, end_point, maze_size, grid_distance, walls, pixel_grid = get_new_training_example(x, network.k, network.sd)
            robot_pos = np.array(start_point, dtype=np.float32)
            end_point = np.array(end_point, dtype=np.float32)
            walls_np, BSP = _build_bsp_tree(walls)
            robot_pos = robot_pos + 0.5
            end_point = end_point + 0.5

            # Initializations
            angle = float((7 * math.pi) / 4)
            total_turn_delta = 0.0
            episode_start = buffered_steps
            RegionsExplored = set()
            PixelVisits = np.zeros(np.asarray(pixel_grid).shape)    # how often each pixel has been entered this episode
            stalled = False

            start_region = goal_region = None
            if grid_distance > 0:
                start_region = tuple((np.floor(robot_pos / grid_distance) * grid_distance).tolist())
                goal_region = tuple((np.floor(end_point / grid_distance) * grid_distance).tolist())
                RegionsExplored.add(start_region)

            network.EnergyUsed = 0.0
            robot_path = [robot_pos.copy()]
            episode_reward = 0.0
            episode_length = 0
            maze_solved = False
            termination = "time_limit"
            skipped = False
            max_ray = ray_max_distance(maze_size, VARRayDistance)

            view.update(
                grid=np.asarray(pixel_grid), rows=pixel_grid.shape[0], cols=pixel_grid.shape[1], walls=walls,
                grid_distance=int(grid_distance), maze_size=maze_size, x=x, episode=episode_count + 1,
                episode_id=view.episode_id + 1, end_point=end_point, start_region=start_region,
                goal_region=goal_region, explored=RegionsExplored, path=robot_path, step=0, rays=None,
                last_move=None, collision=False, dead_end=False, region=None, max_ray=max_ray, banner=None,
                episode_reward=0.0, reward=0.0, step_size=0.0, turn_deg=0.0, success_rate=success_rate,
                past=past, not_past=not_past, buffered=buffered_steps, updates=updates, finished=False,
            )

            for step in range(network.MaxSteps):
                done = False
                State = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size, end_point)
                distance_finish = float(np.linalg.norm(end_point - robot_pos))

                # ---- show this step: robot + the rays it just measured
                rays = lidar_rays(State, float(network.RayOffset.detach().item()), angle, int(VARNumberOfRayCasts), network.FOV, max_ray)
                cast_lengths = [r[1] for r in rays if r[2]]
                view.update(
                    pos=robot_pos.copy(), angle=angle, rays=rays, step=step, goal_distance=distance_finish,
                    rays_cast=network.last_rays_cast, lidar_ms=0.0,
                    min_ray=min(cast_lengths) if cast_lengths else None, energy_used=float(network.EnergyUsed),
                )
                yield "sense"
                if ctl.skip:                                # S key / curriculum jump: throw this episode away
                    ctl.skip = False
                    skipped = True
                    break

                # Get action
                if deterministic:
                    with torch.no_grad():
                        mean, _stds, value = network.forward(State)
                        Action = network._squash_action(mean).cpu().numpy()
                    log_prob = LatentAction = None
                else:
                    Action, log_prob, value, LatentAction, _ = network.get_action_and_value(State)

                # params (old angle, angle, etc.)
                old_angle = float(angle)
                new_angle, step_size = float(Action[0]), float(Action[1])
                # turn limit: no limit at success_rate 0, tightens to target_max_turn as the success rate rises
                turn_limit = math.pi - (math.pi - VARtargetMaxTurn) * min(1.0, ramp_rate / TURN_LIMIT_SUCCESS)
                turn = (new_angle - old_angle + math.pi) % (2 * math.pi) - math.pi
                turn_excess = max(0.0, abs(turn) - turn_limit)    # how far past the limit the policy asked to turn
                turn = max(-turn_limit, min(turn_limit, turn))
                new_angle = (old_angle + turn + math.pi) % (2 * math.pi) - math.pi

                # Virtually update robot stats without directional floor bias.
                new_pos = (robot_pos + step_size * np.array([np.cos(new_angle), np.sin(new_angle)])).astype(np.float32)
                visit_pixel = tuple(int(np.clip(np.floor(new_pos[i]), 0, PixelVisits.shape[i] - 1)) for i in range(2))
                current_pixel_visits = PixelVisits[visit_pixel]         # times this pixel was entered before now
                PixelVisits[visit_pixel] += 1
                robot_path.append(new_pos.copy())
                angle_diff_step = abs((new_angle - old_angle + math.pi) % (2 * math.pi) - math.pi)
                total_turn_delta += angle_diff_step # turn sharpness

                # Get state info
                run_into_wall = bool(RunIntoWall(robot_pos, walls, new_pos[0] - robot_pos[0], new_pos[1] - robot_pos[1]))
                in_dead_end, region_pos = network.InDeadEnd(grid_distance, new_pos, walls_np)
                region_tuple = tuple(region_pos.tolist())

                if region_tuple == start_region or region_tuple == goal_region:
                    in_dead_end = False
                region_unexplored = region_tuple not in RegionsExplored
                if region_unexplored:
                    RegionsExplored.add(region_tuple)

                # Get reward
                next_distance_finish = float(np.linalg.norm(end_point - new_pos))
                distance_progress = distance_finish - next_distance_finish
                in_goal_pixel = bool(np.all(np.floor(new_pos / grid_distance) == np.floor(end_point / grid_distance)))
                goal_distance = 0.0 if in_goal_pixel else next_distance_finish

                # nearest wall the LiDAR saw this step, in pixels (rays outside the FOV hold the max distance, so they never win)
                min_ray_px = min(State[:int(VARNumberOfRayCasts)]) * max_ray

                # called by keyword, so it keeps working however reward()'s parameters are ordered; the optional ones are
                # only passed if reward() has them
                reward_kwargs = dict(
                    distance_progress=distance_progress, run_into_wall=run_into_wall, in_dead_end=in_dead_end,
                    distance_finish=goal_distance, region_unexplored=region_unexplored, robot_pos=robot_pos,
                    new_pos=new_pos, step_size=step_size, success_rate=ramp_rate, angle=old_angle,
                    new_angle=new_angle, maze_size=maze_size, done=done, turn_excess=turn_excess,
                    min_ray_px=min_ray_px, VARRayDistance=VARRayDistance, revisits=current_pixel_visits,
                )
                Reward, done, maze_solved = network.reward(**{k: v for k, v in reward_kwargs.items() if k in reward_params})

                # end hopeless episodes early: almost no net movement over the last STALL_WINDOW steps
                if not done and len(robot_path) > STALL_WINDOW and \
                        np.linalg.norm(new_pos - robot_path[-1 - STALL_WINDOW]) < STALL_MIN_DISPLACEMENT:
                    Reward -= STALL_END_PENALTY
                    done, stalled = True, True
                episode_reward += Reward
                episode_length += 1

                # store episode data (training only)
                if train:
                    network.store_transition(State, LatentAction, Reward, log_prob, done, value, buffered_steps)
                    buffered_steps += 1

                # ---- publish the result of this step (drawn next frame)
                view.update(
                    last_move=(robot_pos.copy(), new_pos.copy()), collision=run_into_wall,
                    dead_end=bool(in_dead_end), region=region_tuple, turn_deg=math.degrees(turn),
                    turn_limit_deg=math.degrees(turn_limit), step_size=step_size, reward=float(Reward),
                    episode_reward=float(episode_reward), energy_used=float(network.EnergyUsed),
                    value=float(value), buffered=buffered_steps,
                )

                # update the robot (angle / position)
                robot_pos, angle = new_pos, new_angle
                if done:
                    termination = "success" if maze_solved else ("collision" if run_into_wall else ("stalled" if stalled else "dead_end"))
                    break

            if skipped:
                buffered_steps = episode_start              # drop the half-finished episode from the PPO buffer
                continue

            past += int(maze_solved)
            not_past += int(not maze_solved)
            success_rate = past / (past + not_past)

            recent_outcomes.append(int(maze_solved))
            rolling_success = sum(recent_outcomes) / len(recent_outcomes)
            if len(recent_outcomes) == CURRICULUM_WINDOW:
                ramp_rate = max(ramp_rate, rolling_success)
            if CURRICULUM_ENABLED:   # promote / demote on the success rate of the last CURRICULUM_WINDOW episodes at this stage
                stage_outcomes.append(int(maze_solved))
                if len(stage_outcomes) == CURRICULUM_WINDOW:
                    stage_rate = sum(stage_outcomes) / CURRICULUM_WINDOW
                    if stage_rate >= CURRICULUM_PROMOTE and stage < len(CURRICULUM_STAGES) - 1:
                        stage += 1
                        stage_outcomes.clear()
                    elif stage_rate <= CURRICULUM_DEMOTE and stage > 0:
                        stage -= 1
                        stage_outcomes.clear()
                    x = CURRICULUM_STAGES[stage]
            episode_count += 1

            view.update(
                pos=robot_pos.copy(), angle=angle, rays=None, goal_distance=float(np.linalg.norm(end_point - robot_pos)),
                success_rate=success_rate, past=past, not_past=not_past, x=x, stage=stage,
                rolling_success=rolling_success, rolling_n=len(recent_outcomes), ramp_rate=ramp_rate,
                banner=(termination.replace("_", " ").upper(), OUTCOME_COLOURS[termination]),
            )
            view.recent.append(termination)
            view.recent_steps.append(episode_length)
            yield "episode_end"

            if not train:
                continue

            # Comput GAE and PPO / other opitimizations
            if done:
                next_value = torch.tensor(0.0, dtype=torch.float32)
            else:
                with torch.no_grad():
                    FinalState = network.GetState(robot_pos, VARNumberOfRayCasts, BSP, network.RayOffset, network.FOV, angle, VARRayDistance, maze_size, end_point)
                    _, _, next_value = network.forward(FinalState)

            episode_slice = slice(episode_start, buffered_steps)
            advantages, returns = _compute_gae(network.rewards[episode_slice], network.values[episode_slice],
                                              network.dones[episode_slice], next_value)
            buffered_advantages.append(advantages)
            buffered_returns.append(returns)
            buffer_turn_total += total_turn_delta
            is_last = rollouts is not None and _rollout == rollouts - 1
            rollout_target = MIN_ROLLOUT_STEPS_LEARNING if sum(recent_outcomes) > 0 else MIN_ROLLOUT_STEPS
            if buffered_steps < rollout_target and not is_last:
                view.update(buffered=buffered_steps, rollout_target=rollout_target)
                continue

            # TRAINING PPO LOOP
            view.update(banner=("PPO UPDATE ...", DIM), buffered=buffered_steps)
            yield "update"
            diagnostics, final_kl, kl_stopped, explained_variance = _ppo_update(
                network, optimizer, ppo_parameters, network.states[:buffered_steps], network.actions[:buffered_steps],
                network.log_probs[:buffered_steps], torch.cat(buffered_returns), torch.cat(buffered_advantages))
            updates += 1

            # Update weights adam
            if w_optimizer is not None:
                avg_turn = buffer_turn_total / buffered_steps
                w_turn_loss = -network.w_turn * (avg_turn - network.target_max_turn)
                w_optimizer.zero_grad()
                w_turn_loss.backward()
                w_optimizer.step()
                with torch.no_grad():
                    network.log_w_turn.clamp_(math.log(1e-4), math.log(10.0))

            network.clear_memory(capacity) # return to next episode
            network.set_max_step(ramp_rate) # buffer is empty, so it is safe to change the step cap now
            buffered_steps = 0
            buffered_advantages, buffered_returns = [], []
            buffer_turn_total = 0.0
            view.update(banner=None, buffered=0, updates=updates, last_kl=float(final_kl))

    # ------------------------------------------------------------ window loop
    pygame.init()
    pygame.display.set_caption(f"Maze PPO viewer - {test} - {mode}")
    screen = pygame.display.set_mode(window, pygame.RESIZABLE)
    fonts = SimpleNamespace(small=pygame.font.SysFont(MONO, 14), big=pygame.font.SysFont(MONO, 22, bold=True))
    ui = SimpleNamespace(
        paused=False, turbo=False, speed=min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - steps_per_second)),
        show_rays=True, show_trail=True, show_grid=True, show_segments=False, show_regions=True, show_help=True,
        shot=False, layout=None,
    )
    cache = {}
    stream = episodes()
    state = SimpleNamespace(finished=False, hold=0.0)

    def advance():
        if state.finished:
            return "finished"
        try:
            return next(stream)
        except StopIteration:
            state.finished = True
            view.finished = True
            return "finished"

    clock = pygame.time.Clock()
    budget, frames, shots, step_requests, running = 0.0, 0, 0, 0, True
    try:
        advance()                                           # fill the view before the first frame
        while running:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif key == pygame.K_SPACE:
                        ui.paused = not ui.paused
                    elif key in (pygame.K_RIGHT, pygame.K_n):
                        ui.paused, step_requests = True, step_requests + 1
                    elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                        ui.speed = min(len(SPEEDS) - 1, ui.speed + 1)
                    elif key in (pygame.K_DOWN, pygame.K_MINUS):
                        ui.speed = max(0, ui.speed - 1)
                    elif key == pygame.K_f:
                        ui.turbo = not ui.turbo
                    elif key == pygame.K_s:
                        ctl.skip = True
                        state.hold = 0.0                    # do not sit through the end-of-episode pause
                        step_requests += int(ui.paused)     # while paused, take one step so the new maze shows
                    elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        if key == pygame.K_RIGHTBRACKET:
                            ctl.set_x = next((s for s in stages if s > view.x), stages[-1])
                        else:
                            ctl.set_x = next((s for s in reversed(stages) if s < view.x), stages[0])
                        ctl.skip = True
                        state.hold = 0.0
                        step_requests += int(ui.paused)
                    elif key == pygame.K_l:
                        ui.show_rays = not ui.show_rays
                    elif key == pygame.K_t:
                        ui.show_trail = not ui.show_trail
                    elif key == pygame.K_g:
                        ui.show_grid = not ui.show_grid
                    elif key == pygame.K_w:
                        ui.show_segments = not ui.show_segments
                    elif key == pygame.K_e:
                        ui.show_regions = not ui.show_regions
                    elif key == pygame.K_h:
                        ui.show_help = not ui.show_help
                    elif key == pygame.K_p:
                        ui.shot = True

            # ---- advance the simulation
            if step_requests:
                for _ in range(step_requests):
                    advance()
                step_requests = 0
            elif not ui.paused and not state.finished:
                if ui.turbo:
                    stop_at = time.perf_counter() + 0.03
                    while time.perf_counter() < stop_at:
                        if advance() in ("update", "finished"):
                            break
                elif state.hold > 0:
                    state.hold -= dt
                else:
                    budget += dt * SPEEDS[ui.speed]
                    n = min(int(budget), 200)
                    budget = min(budget - n, 1.0)
                    for _ in range(n):
                        event_name = advance()
                        if event_name == "episode_end":
                            state.hold = HOLD_SECONDS
                            break
                        if event_name in ("update", "finished"):
                            break

            # ---- draw
            screen = pygame.display.get_surface()
            draw_frame(screen, fonts, view, ui, cache)
            if on_frame is not None:
                on_frame(screen, view, ui)
            if ui.shot:
                ui.shot, shots = False, shots + 1
                Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(Path(screenshot_dir) / f"viewer_{time.strftime('%Y%m%d_%H%M%S')}_{shots:03d}.png"))
            pygame.display.flip()
            frames += 1
            if max_frames is not None and frames >= max_frames:
                running = False
    finally:
        pygame.quit()
    return network


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch the PPO maze robot in pygame.")
    parser.add_argument("--policy", help="Policy_*.pt to watch (no training)")
    parser.add_argument("--start-x", type=int, default=0, help="curriculum counter to start from (51+ skips the empty arenas)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sps", type=int, default=15, help="steps per second")
    parser.add_argument("--sampled", action="store_true", help="when watching a policy, sample actions instead of using the mean")
    args = parser.parse_args()
    StartViewer("viewer", 48, 24, 24, 9000, 34, "orth", 0.5, 1, 180, None, seed=args.seed, policy_path=args.policy,
                deterministic=(False if args.sampled else None), start_difficulty=args.start_x, steps_per_second=args.sps)
    sys.exit(0)