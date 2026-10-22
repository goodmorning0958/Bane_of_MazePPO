from stats_analytics import logistic_regression, interactions_logistic_regression
import statsmodels as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('results.csv')

required_columns = ['ray_count', 'fov', 'sensing_distance', 'energy_constraint', 'turn_sharpness', 'success']
assert all(col in data.columns for col in required_columns)

configurations_groups = data.groupby(['ray_count', 'fov', 'sensing_distance', 'energy_constraint', 'turn_sharpness'], as_index=False).agg(success_rate='success', 'mean')

configurations_groups['success_rate'] *= 100

converting_to_read = [('ray_count', 'Ray Count'), ('fov', 'Field of View (Degrees)'), ('sensing_distance', 'Sensing Distance'), ('energy_constraint', 'Energy Constraint'), ('turn_sharpness', 'Turn Sharpness')]
for column, readable_name in converting_to_read:
    plt.figure(figsize=(8, 5))
    plt.scatter(configurations_groups[column], configurations_groups['success_rate'])
    plt.xlabel(readable_name)
    plt.ylabel("Success Rate (%)")
    plt.title(f"Success Rate vs. {readable_name]")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.4)
    plt.tight_layout

    filename = f"success_rate_vs_{column}.png"
    plt.savefig(filename, dpi=300)

    plt.show()
    plt.close()

result = logistic_regression(data)

averages = data[['ray_count', 'fov', 'sensing_distance', 'energy_constraint', 'turn_sharpness']].mean()

for column, readable_name in converting_to_read:
    x_axis = np.linspace(data[column].min(), data[column].max(), 100)
    predictions = pd.DataFrame({'ray_count': averages['ray_count'], 'fov': averages['fov'], 'sensing_distance': averages['sensing_distance'], 'energy_constraint': averages['energy_constraint'], 'turn_sharpness': averages['turn_sharpness']}, index=range(len(x_axis)))
    predictions[column] = x_axis

    prediction_X = sm.add_constant(predictions, has_constant='add')
    probablities = result.predict(prediction_X)

    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, probablities*100)
    plt.title(f"Predicted Success Rate vs. {readable_name}")
    plt.xlabel(readable_name)
    plt.ylabel("Predicted Success Rate (%)")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.4)
    plt.tight_layout

    filename = f"predicted_success_rate_vs_{column}.png"
    plt.savefig(filename, dpi=300)

    plt.show()
    plt.close()