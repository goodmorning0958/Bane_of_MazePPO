from stats_analytics import best_compromise, pareto_front
import pandas as pd
import glob
import os

files = glob.glob('policy_eval_results/agent_*/EpisodeData_eval_sweep_*.csv')

all_data = []

for file in files:
    dataa = pd.read_csv(file)

    agent_id = os.path.basename(file).replace('EpisodeData_eval_sweep_', '').replace('.csv', '')
    dataa['agent_id'] = int(agent_id)

    all_data.append(dataa)

dataaa = pd.concat(all_data, ignore_index=True)

weightss = {
    0: {'success': 1, 'path_efficiency': 0, 'energy': 0, 'rays': 0},
    1: {'success': 0.5, 'path_efficiency': 0.2, 'energy': 0.15, 'rays': 0.15},
    2: {'success': 0.25, 'path_efficiency': 0.25, 'energy': 0.25, 'rays': 0.25},
    3: {'success': 0.2, 'path_efficiency': 0.1, 'energy': 0.35, 'rays': 0.35},
    4: {'success': 0.3, 'path_efficiency': 0.1, 'energy': 0.1, 'rays': 0.5},
    5: {'success': 0, 'path_efficiency': 0, 'energy': 1, 'rays': 0},
}

print("pareto front results:\n")
for i in range(6):
    weights = weightss[i]
    table = best_compromise(dataaa, weights)

    print(f"weights: {weights}\n")
    print(f"{table}\n\n")

print("pareto front results:\n")
patero_frontt = pareto_front(dataaa)
print(patero_frontt)
