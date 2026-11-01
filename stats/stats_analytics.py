from scipy.stats import qmc
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import statsmodels.api as sm
import pandas as pd

def precision_analysis(p, e):
    n = np.ceil((1.96**2*p*(1-p))/e**2)
    return n

def latin_hypercube_sampling(samples):
    latin_hypercube = qmc.LatinHypercube(d=4, seed=68)
    sampling = latin_hypercube.random(samples)

    ray_count = np.round(qmc.scale(sampling[:, 0:1], [4], [128])).astype(int).flatten()
    fov = qmc.scale(sampling[:, 1:2], [45], [360]).flatten()
    energy_constraint = qmc.scale(sampling[:, 2:3], [30], [100]).flatten()
    turn_sharpness = qmc.scale(sampling[:, 3:4], [15], [360]).flatten()
    
    lhs_configurations = []
    for i in range(samples):
        lhs_configurations.append({'ray_count': ray_count[i], 'fov': fov[i], 'energy_constraint': energy_constraint[i], 'turn_sharpness': turn_sharpness[i]})

    return lhs_configurations

def logistic_regression(data):
    X = data[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']].copy()
    X = sm.add_constant(X)

    y = data['success']

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    return result

def linear_regression(data, dtype):
    mask = data['success'] == 1
    successful = data[mask]

    X = successful[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']]
    X = sm.add_constant(X)
    y = np.log(successful[dtype])

    model = sm.OLS(y, X)
    result = model.fit()

    return result

def extra_stats(result):
    variable_and_increments = {'VARNumberOfRayCasts': 8, 'VARFOV': 90, 'VARAllowedEnergy': 30, 'VARtargetMaxTurn': 0.5}

    print("\nOdds Ratios:")
    for variable, increment in variable_and_increments.items():
        odds_ratio = np.exp(result.params[variable]*increment)
        print(f"\n{variable}: {odds_ratio:.5f} (per {increment} increase in {variable})")

def roc_auc(result, data):
    X = data[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']]
    X = sm.add_constant(X)

    y = data['success']
    predicted_probability = result.predict(X)

    auc = roc_auc_score(y, predicted_probability)
    fpr, tpr, thresholds = roc_curve(y, predicted_probability)

    print(f"ROC-AUC: {auc:.3f}")

    return fpr, tpr, thresholds, auc

def roc_auc_for_interactions(result, data):
    X = data[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']].copy()

    X['VARNumberOfRayCasts'] -= X['VARNumberOfRayCasts'].mean()
    X['VARFOV'] -= X['VARFOV'].mean()
    X['VARAllowedEnergy'] -= X['VARAllowedEnergy'].mean()
    X['VARtargetMaxTurn'] -= X['VARtargetMaxTurn'].mean()

    X['ray_count_fov'] = X['VARNumberOfRayCasts'] * X['VARFOV']
    X['energy_constraint_turn_sharpness'] = X['VARAllowedEnergy'] * X['VARtargetMaxTurn']
    X['ray_count_energy_constraint'] = X['VARNumberOfRayCasts'] * X['VARAllowedEnergy']
    X['fov_turn_sharpness'] = X['VARFOV'] * X['VARtargetMaxTurn']
    X['ray_count_turn_sharpness'] = X['VARNumberOfRayCasts'] * X['VARtargetMaxTurn']
    X['fov_energy_constraint'] = X['VARFOV'] * X['VARAllowedEnergy']

    X = sm.add_constant(X)

    y = data['success']
    predicted_probability = result.predict(X)

    auc = roc_auc_score(y, predicted_probability)
    fpr, tpr, thresholds = roc_curve(y, predicted_probability)

    print(f"ROC-AUC: {auc:.3f}")

    return fpr, tpr, thresholds, auc

def interactions_logistic_regression(data):
    X = data[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']].copy()

    X['VARNumberOfRayCasts'] -= X['VARNumberOfRayCasts'].mean()
    X['VARFOV'] -= X['VARFOV'].mean()
    X['VARAllowedEnergy'] -= X['VARAllowedEnergy'].mean()
    X['VARtargetMaxTurn'] -= X['VARtargetMaxTurn'].mean()
    
    X['ray_count_fov'] = X['VARNumberOfRayCasts'] * X['VARFOV']
    X['energy_constraint_turn_sharpness'] = X['VARAllowedEnergy'] * X['VARtargetMaxTurn']
    X['ray_count_energy_constraint'] = X['VARNumberOfRayCasts'] * X['VARAllowedEnergy']
    X['fov_turn_sharpness'] = X['VARFOV'] * X['VARtargetMaxTurn']
    X['ray_count_turn_sharpness'] = X['VARNumberOfRayCasts'] * X['VARtargetMaxTurn']
    X['fov_energy_constraint'] = X['VARFOV'] * X['VARAllowedEnergy']

    X = sm.add_constant(X)
    y = data['success']

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    return result

def bootstrap_success_rate(data, iterations = 10000, seed = 42):
    rng = np.random.default_rng(seed)
    groups = {agent_id: group['success'].to_numpy() for agent_id, group in data.groupby('agent_id')}
    agent_ids = list(groups)

    bootstrap_rates = []

    for i in range(iterations):
        bootstrap_samples = rng.choice(agent_ids, size = len(agent_ids), replace = True)
        bootstrap_sample = np.concatenate([groups[agent_id] for agent_id in bootstrap_samples])
        bootstrap_rates.append(bootstrap_sample.mean())

    bootstrap_rates = np.asarray(bootstrap_rates)

    lower_percentitle = np.percentile(bootstrap_rates, 2.5)
    upper_percentitle = np.percentile(bootstrap_rates, 97.5)

    return bootstrap_rates.mean(), lower_percentitle, upper_percentitle

def bootstrap_odds_ratios(data, iterations = 1000, seed = 42):
    variables = ['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']
    variable_and_increments = {'VARNumberOfRayCasts': 8, 'VARFOV': 90, 'VARAllowedEnergy': 30, 'VARtargetMaxTurn': 0.5}
    bootstrap_ratios = []
    successful_models = 0
    rng = np.random.default_rng(seed)
    groups = {agent_id: group for agent_id, group in data.groupby('agent_id')}
    agent_ids = list(groups)

    for i in range(iterations):
        bootstrap_sample = rng.choice(agent_ids, size = len(agent_ids), replace = True)
        bootstrap_samples = pd.concat([groups[agent_id] for agent_id in bootstrap_sample], ignore_index = True)

        X = sm.add_constant(bootstrap_samples[variables])
        y = bootstrap_samples['success']

        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=0)

            odds_ratios = {}
            for variable, increment in variable_and_increments.items():
                odds_ratio = np.exp(result.params[variable]*increment)
                odds_ratios[variable] = odds_ratio

            bootstrap_ratios.append([odds_ratios[variable] for variable in variables])

            successful_models += 1
        except Exception:
            continue

    if successful_models == 0:
        raise ValueError("No models suceeded")

    print(f"Successful Models: {successful_models}/{iterations}")

    bootstrap_ratios = np.asarray(bootstrap_ratios)

    lower_percentitle = np.percentile(bootstrap_ratios, 2.5, axis=0)
    upper_percentitle = np.percentile(bootstrap_ratios, 97.5, axis=0)

    return {variable: {'lower_percentitle': lower, 'upper_percentitle': upper} for variable, lower, upper in zip(variables, lower_percentitle, upper_percentitle)}