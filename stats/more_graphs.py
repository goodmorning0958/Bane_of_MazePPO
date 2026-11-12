import pandas as pd
total_episodes = 100000

averages = {
    "24":0, "16":0, "8":0,
    "360":0, "180":0, "90":0,
    "30":0, "60":0, "150":0,
    "0.5":0, "1.0":0, "3.14159":0,
}
counts = {k: 0 for k in averages}
not_reached = {k: 0 for k in averages}
conditions = pd.read_csv("training_results/manifest.csv")

rays = conditions['VARNumberOfRayCasts'].tolist()
FOV = conditions['VARFOV'].tolist()
energy = conditions['VARAllowedEnergy'].tolist()
turn_sharpness = conditions['VARtargetMaxTurn'].tolist()

for agnt in range(81):
    rayy = rays[agnt]
    FOVV = FOV[agnt]
    energyy = energy[agnt]
    turn_sharpnesss = turn_sharpness[agnt]

    agnt_string = f"{agnt:03d}"
    path = "training_results/agent_" + agnt_string + f"_seed_42/EpisodeData_sweep_{agnt}.csv"
    data = pd.read_csv(path, usecols = ['episode', 'stage'])

    reached = data.loc[data['stage'] >= 5, 'episode']
    episode_at_stage = int(reached.iloc[0]) if len(reached) > 0 else total_episodes

    for key in [f"{rayy}", f"{FOVV}", f"{energyy}", f"{turn_sharpnesss}"]:
        averages[key] += episode_at_stage
        counts[key] += 1
        if len(reached) == 0:
            not_reached[key] += 1

averages = {k: v / counts[k] for k, v in averages.items()}
for k in averages:
    print(f"{k:>8}: {averages[k]:9.0f} episodes to stage 5 ({counts[k]} agents, {not_reached[k]} never reached it)")