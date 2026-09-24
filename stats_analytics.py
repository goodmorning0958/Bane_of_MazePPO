from scipy.stats import qmc
import numpy as np
import statsmodels.api as sm
import pandas as pd

def precision_analysis(p, e):
    n = np.round((1.96**2*p*(1-p))/e**2).astype(int)
    return n

def latin_hypercube_sampling(samples):
    latin_hypercube = qmc.LatinHypercube(d=3, seed=68)
    sampling = latin_hypercube.random(samples)

    ray_count = np.round(qmc.scale(sampling[:, 0:1], [4], [128])).astype(int).flatten()
    fov = qmc.scale(sampling[:, 1:2], [45], [360]).flatten()
    distance = qmc.scale(sampling[:, 2:3], [0.5], [5]).flatten()

    lhs_configurations = []
    for i in range(samples):
        lhs_configurations.append({'ray_count': ray_count[i], 'fov': fov[i], 'sensing_distance': distance[i]})

    return lhs_configurations

def logistic_regression(data):
    X = data[['ray_count', 'fov', 'sensing_distance']]
    X = sm.add_constant(X)

    y = data['success']

    model = sm.Logit(y, X)
    result = model.fit()

    return result