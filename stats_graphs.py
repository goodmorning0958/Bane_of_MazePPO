from stats_analytics import logistic_regression, interactions_logistic_regression
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('manifest.csv')

required_columns = ['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn', 'final_success_binary']
assert all(col in data.columns for col in required_columns)

configurations_groups = data.groupby(['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn'], as_index=False).agg(success_rate=('final_success_binary', 'mean'))

configurations_groups['success_rate'] *= 100

converting_to_read = [('VARNumberOfRayCasts', 'Ray Count'), ('VARFOV', 'Field of View (Degrees)'), ('VARAllowedEnergy', 'Allowed Energy'), ('VARtargetMaxTurn', 'Turn Sharpness')]
for column, readable_name in converting_to_read:
    plt.figure(figsize=(8, 5))
    plt.scatter(configurations_groups[column], configurations_groups['success_rate'])
    plt.xlabel(readable_name)
    plt.ylabel("Success Rate (%)")
    plt.title(f"Success Rate vs. {readable_name}")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.4)
    plt.tight_layout

    filename = f"success_rate_vs_{column}.png"
    plt.savefig(filename, dpi=300)

    plt.show()
    plt.close()

result = logistic_regression(data)

averages = data[['VARNumberOfRayCasts', 'VARFOV', 'VARAllowedEnergy', 'VARtargetMaxTurn']].mean()

for column, readable_name in converting_to_read:
    x_axis = np.linspace(data[column].min(), data[column].max(), 100)
    predictions = pd.DataFrame({'VARNumberOfRayCasts': averages['VARNumberOfRayCasts'], 'VARFOV': averages['VARFOV'], 'VARAllowedEnergy': averages['VARAllowedEnergy'], 'VARtargetMaxTurn': averages['VARtargetMaxTurn']}, index=range(len(x_axis)))
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
    plt.tight_layout()

    filename = f"predicted_success_rate_vs_{column}.png"
    plt.savefig(filename, dpi=300)

    plt.show()
    plt.close()