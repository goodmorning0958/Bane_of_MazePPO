from pathlib import Path
import shutil
import sys

root = Path.cwd()
crlf = "\r\n"

moves = [
    ("testing/main.py", "model/main.py"),
    ("model/Policy_tester.py", "testing/Policy_tester.py"),
    ("environment/physics_calibrator", "environment/physics_calibrator.py"),
]

edits = {
    "model/main.py": [
        ("from agent_ppo import StartAgent" + crlf, "from model.agent_ppo import StartAgent" + crlf),
        ("import actor_critic" + crlf, "from model import actor_critic" + crlf),
        ("import agent_ppo" + crlf, "from model import agent_ppo" + crlf),
        ("import PPO_action_converter" + crlf, "from environment import PPO_action_converter" + crlf),
        ('OUT = Path("runs") / "sweep_81_v5"', 'OUT = Path("training_results_v5")'),
    ],
    "testing/Policy_tester.py": [
        ("from training_example_generator import get_new_training_example", "from environment.training_example_generator import get_new_training_example"),
        ("from A_star_search import Normal_A_star_search", "from environment.A_star_search import Normal_A_star_search"),
        ("from RunIntoWall import RunIntoWall", "from environment.RunIntoWall import RunIntoWall"),
        ("from actor_critic import (", "from model.actor_critic import ("),
        ("from maze_utils import _build_bsp_tree", "from environment.maze_utils import _build_bsp_tree"),
        ("from energy_lookup_table import move_joules", "from environment.energy_lookup_table import move_joules"),
    ],
    "testing/policy_tester_main.py": [
        ("from Policy_tester import StartAgent", "from testing.Policy_tester import StartAgent"),
        ('policies = Path("runs") / "sweep_81_v5"', 'policies = Path("training_results_v5")'),
        ('out = Path("runs") / "eval_81_v5"', 'out = Path("policy_eval_results")'),
    ],
    "testing/diagnostics.py": [
        ("from training_example_generator import get_new_training_example", "from environment.training_example_generator import get_new_training_example"),
        ("from maze_utils import _build_bsp_tree, raycaster_env", "from environment.maze_utils import _build_bsp_tree, raycaster_env"),
        ("from RunIntoWall import RunIntoWall", "from environment.RunIntoWall import RunIntoWall"),
        ("from PPO_action_converter import ConvertRobotAction", "from environment.PPO_action_converter import ConvertRobotAction"),
        ("    from actor_critic import ActorCritic", "    from model.actor_critic import ActorCritic"),
    ],
    "model/actor_critic.py": [
        ("from PPO_action_converter import ConvertRobotAction", "from environment.PPO_action_converter import ConvertRobotAction"),
        ("from maze_utils import raycaster_env, _fov_to_radians", "from environment.maze_utils import raycaster_env, _fov_to_radians"),
    ],
    "model/agent_ppo.py": [
        ("from training_example_generator import get_new_training_example", "from environment.training_example_generator import get_new_training_example"),
        ("from A_star_search import Normal_A_star_search", "from environment.A_star_search import Normal_A_star_search"),
        ("from RunIntoWall import RunIntoWall", "from environment.RunIntoWall import RunIntoWall"),
        ("from agent_viewer import Dashboard" + crlf, ""),
        ("from actor_critic import (", "from model.actor_critic import ("),
        ("from maze_utils import raycaster_env, _fov_to_radians, _build_bsp_tree", "from environment.maze_utils import raycaster_env, _fov_to_radians, _build_bsp_tree"),
    ],
    "environment/maze_utils.py": [
        ("from SImulated_BSP_LiDAR import Tree", "from environment.SImulated_BSP_LiDAR import Tree"),
    ],
    "environment/energy_lookup_table.py": [
        ('"runs/physics_calibration/calibration_moves.csv"', '"environment/physics_calibration/calibration_moves.csv"'),
    ],
    "environment/physics_calibrator.py": [
        ("import physics" + crlf, "from environment import physics" + crlf),
        ('OUT = Path("runs") / "mujoco_calibration"', 'OUT = Path("environment") / "physics_calibration"'),
        ('mujoco.MjModel.from_xml_path("model.xml")', 'mujoco.MjModel.from_xml_path("environment/model.xml")'),
    ],
    ".gitignore": [
        ("*/__pycache__/", "__pycache__/" + crlf + "training_results_v5/" + crlf),
    ],
}

old_results = root / "training_results v5"
new_results = root / "training_results_v5"

problems = []
for folder in ["environment", "model", "testing"]:
    if not (root / folder).is_dir():
        problems.append(f"missing folder {folder}")
if not old_results.is_dir():
    problems.append('missing folder "training_results v5"')
if new_results.exists():
    problems.append("training_results_v5 already exists")
for src, dst in moves:
    if not (root / src).is_file():
        problems.append(f"missing {src}")
    if (root / dst).exists():
        problems.append(f"{dst} already exists")
for path in ["testing/agent_viewer.py", "real_mazes.json", "____REASEARCH_PAPER____.txt"]:
    if not (root / path).is_file():
        problems.append(f"missing {path}")

current = {dst: src for src, dst in moves}
new_text = {}
for path, changes in edits.items():
    source = root / current.get(path, path)
    if not source.is_file():
        problems.append(f"missing {source.relative_to(root)}")
        continue
    with open(source, encoding="utf-8", newline="") as f:
        text = f.read()
    for old, new in changes:
        if text.count(old) != 1:
            problems.append(f"{source.relative_to(root)}: expected to find {old.strip()!r} once, found {text.count(old)}")
        text = text.replace(old, new)
    new_text[path] = text

if problems:
    print("nothing was changed, because:")
    for p in problems:
        print("  " + p)
    sys.exit(1)

old_results.rename(new_results)
for src, dst in moves:
    (root / src).rename(root / dst)
for path, text in new_text.items():
    with open(root / path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
for folder in ["environment", "model", "testing"]:
    (root / folder / "__init__.py").touch()
(root / "testing" / "agent_viewer.py").unlink()
(root / "old").mkdir(exist_ok=True)
shutil.move(str(root / "real_mazes.json"), str(root / "old" / "real_mazes.json"))
(root / "docs").mkdir(exist_ok=True)
shutil.move(str(root / "____REASEARCH_PAPER____.txt"), str(root / "docs" / "____REASEARCH_PAPER____.txt"))
removed = 0
for leftover in [new_results / "failed.txt"] + sorted(new_results.glob("agent_*/error.txt")):
    if leftover.is_file():
        leftover.unlink()
        removed += 1

print("done:")
print('  renamed "training_results v5" to training_results_v5')
for src, dst in moves:
    print(f"  moved {src} to {dst}")
print(f"  fixed imports and paths in {len(new_text)} files")
print("  added __init__.py to environment, model and testing")
print("  deleted testing/agent_viewer.py")
print("  moved real_mazes.json to old/ and the research paper to docs/")
print(f"  deleted {removed} leftover failed.txt / error.txt files")