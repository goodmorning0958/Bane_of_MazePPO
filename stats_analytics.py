from scipy.stats import qmc
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import statsmodels.api as sm
import pandas as pd

def precision_analysis(p, e):
    n = np.round((1.96**2*p*(1-p))/e**2).astype(int)
    return n

def latin_hypercube_sampling(samples):
    latin_hypercube = qmc.LatinHypercube(d=4, seed=68)
    sampling = latin_hypercube.random(samples)

    ray_count = np.round(qmc.scale(sampling[:, 0:1], [4], [128])).astype(int).flatten()
    fov = qmc.scale(sampling[:, 1:2], [45], [360]).flatten()
    energy_constraint = qmc.scale(sampling[:, 3:4], [30], [100]).flatten()
    turn_sharpness = qmc.scale(sampling[:, 4:5], [15], [360]).flatten()
    
    lhs_configurations = []
    for i in range(samples):
        lhs_configurations.append({'ray_count': ray_count[i], 'fov': fov[i], 'energy_constraint': energy_constraint[i], 'turn_sharpness': turn_sharpness[i]})

    return lhs_configurations

def logistic_regression(data):
    X = data[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']]
    X = sm.add_constant(X)

    y = data['final_success_binary']

    model = sm.Logit(y, X)
    result = model.fit()

    return result

def extra_stats(result, data):
    confidence_intervals = result.conf_int()

    odds_ratios = np.exp(result.params)

    success_rate = data['final_success_binary'].mean() * 100

    print("\n95% Confidence Intervals:")
    print(confidence_intervals)

    print("\nOdds Ratios:")
    print(odds_ratios)

    print("\nOverall Success Rate:")
    print(success_rate, "%")

def roc_auc(result, data):
    X = data[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']]
    X = sm.add_constant(X)

    y = data['final_success_binary']
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
    y = data['final_success_binary']

    model = sm.Logit(y, X)
    result = model.fit()

    return result

def bootstrap_sampling(success, iterations=10000):
    success = np.asarray(success)

    bootstrap_rates = []

    for i in range(iterations):
        bootstrap_sample = np.random.choice(success, size=len(success), replace=True)

        bootstrap_rates.append(bootstrap_sample.mean())

    lower_percentitle = np.percentile(bootstrap_rates, 2.5)
    upper_percentitle = np.percentile(bootstrap_rates, 97.5)

    return bootstrap_rates.mean(), lower_percentitle, upper_percentitle