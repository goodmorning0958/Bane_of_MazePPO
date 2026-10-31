from stats_analytics import logistic_regression, interactions_logistic_regression
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

data = pd.read_csv('runs/sweep_81_v5/manifest.csv')

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
    plt.tight_layout()

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

files = glob.glob('runs/sweep_81_v5/*/EpisodeData_seep_*.csv')

all_data = []

for file in files:
    dataa = pd.read_csv(file)

    agent_id = os.path.basename(file).replace('csv', '')
    dataa['agent_id'] = agent_id

    all_data.append(dataa)

training_data = pd.concat(all_data, ignore_index=True)

training_data['episode_group'] = ((training_data['episode'] - 1) // 1000) * 1000

average_success_stuff = training_data.groupby('episode_group')['success'].mean().reset_index()
average_success_stuff['success'] *= 100

plt.figure(figsize=(8, 5))
plt.plot(average_success_stuff['episode_group'], average_success_stuff['success'])
plt.title("Success Rate vs. Amount of Training Episodes Completed")
plt.xlabel("Trainging Episodes Completed")
plt.ylabel("Success Rate")
plt.ylim(-5, 105)
plt.grid(True, alpha=0.4)
plt.tight_layout()

filename = "success_rate_vs_training_episodes.png"
plt.savefig(filename, dpi=300)

plt.show()
plt.close()

training_data['accuracy_group'] = training_data['accuracy'] // 1
accuracy_stuff = training_data.groupby('accuracy_group')['success'].mean().reset_index()
accuracy_stuff['success'] *= 100

plt.figure(figsize=(8,5))
plt.plot(accuracy_stuff['accuracy'], accuracy_stuff['success'])
plt.title("Success Rate vs. Accuracy Compared to A* Path")
plt.xlabel("Accuracy")
plt.ylabel("Success Rate")
plt.ylim(-5, 105)
plt.grid(True, alpha=0.4)
plt.tight_layout()

filename = "success_rate_vs_accuracy.png"
plt.savefig(filename, dpi=300)

plt.show()
plt.close()

training_data['difficulty_group'] = (training_data['difficulty_x'] // 5) * 5
accuracy_for_difficulty = training_data.groupby('difficulty_group')['accuracy'].mean().reset_index()

plt.figure(figsize=(8,5))
plt.plot(accuracy_for_difficulty['difficulty_group'], accuracy_for_difficulty['accuracy'])
plt.title("Accuracy Compared to A* Path vs. Maze Difficulty")
plt.xlabel("Difficulty")
plt.ylabel("Accuracy")
plt.ylim(-5, 105)
plt.grid(True, alpha=0.4)
plt.tight_layout()

filename = "accuracy_vs_maze_difficulty.png"
plt.savefig(filename, dpi=300)

plt.show()
plt.close()