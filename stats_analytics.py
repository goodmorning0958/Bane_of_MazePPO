from scipy.stats import qmc
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import statsmodels.api as sm
import pandas as pd

def precision_analysis(p, e):
    n = np.round((1.96**2*p*(1-p))/e**2).astype(int)
    return n

def latin_hypercube_sampling(samples):
    latin_hypercube = qmc.LatinHypercube(d=5, seed=68)
    sampling = latin_hypercube.random(samples)

    ray_count = np.round(qmc.scale(sampling[:, 0:1], [4], [128])).astype(int).flatten()
    fov = qmc.scale(sampling[:, 1:2], [45], [360]).flatten()
    distance = qmc.scale(sampling[:, 2:3], [0.5], [5]).flatten()
    energy_constraint = qmc.scale(sampling[:, 3:4], [30], [100]).flatten()
    turn_sharpness = qmc.scale(sampling[:, 4:5], [15], [360]).flatten()
    
    lhs_configurations = []
    for i in range(samples):
        lhs_configurations.append({'ray_count': ray_count[i], 'fov': fov[i], 'sensing_distance': distance[i], 'energy_constraint': energy_constraint[i], 'turn_sharpness': turn_sharpness[i]})

    return lhs_configurations

def logistic_regression(data):
    X = data[['ray_count', 'fov', 'sensing_distance', 'energy_constraint', 'turn_sharpness']]
    X = sm.add_constant(X)

    y = data['success']

    model = sm.Logit(y, X)
    result = model.fit()

    return result

def extra_stats(result, data):
    confidence_intervals = result.conf_int()

    odds_ratios = np.exp(result.params)

    success_rate = data['success'].mean() * 100

    print("\n95% Confidence Intervals:")
    print(confidence_intervals)

    print("\nOdds Ratios:")
    print(odds_ratios)

    print("\nOverall Success Rate:")
    print(success_rate, "%")

def roc_auc(result, data):
    X = data[['ray_count', 'fov', 'sensing_distance', 'energy_constraint', 'turn_sharpness']]
    X = sm.add_constant(X)

    y = data['success']
    predicted_probability = result.predict(X)

    auc = roc_auc_score(y, predicted_probability)
    fpr, tpr, thresholds = roc_curve(y, predicted_probability)

    print(f"ROC-AUC: {auc:.3f}")

    return fpr, tpr, thresholds, auc

def interactions_logistic_regression(data):
    X = data[['ray_count', 'fov', 'sensing_distance', 'energy_constraint', 'turn_sharpness']].copy()

    X['ray_count'] -= X['ray_count'].mean()
    X['fov'] -= X['fov'].mean()
    X['sensing_distance'] -= X['sensing_distance'].mean()
    X['energy_constraint'] -= X['energy_constraint'].mean()
    X['turn_sharpness'] -= X['turn_sharpness'].mean()

    X['ray_count_fov'] = X['ray_count'] * X['fov']
    X['ray_count_sensing_distance'] = X['ray_count'] * X['sensing_distance']
    X['fov_sensing_distance'] = X['fov'] * X['sensing_distance']

    X['energy_constraint_turn_sharpness'] = X['energy_constraint'] * X['turn_sharpness']
    X['ray_count_energy_constraint'] = X['ray_count'] * X['energy_constraint']
    X['fov_turn_sharpness'] = X['fov'] * X['turn_sharpness']
    X['sensing_distance_turn_sharpness'] = X['sensing_distance'] * X['turn_sharpness']
    X['ray_count_turn_sharpness'] = X['ray_count'] * X['turn_sharpness']
    X['fov_energy_constraint'] = X['fov'] * X['energy_constraint']
    X['sensing_distance_energy_constraint'] = X['sensing_distance'] * X['energy_constraint']

    X = sm.add_constant(X)
    y = data['success']

    model = sm.Logit(y, X)
    result = model.fit()

    return result