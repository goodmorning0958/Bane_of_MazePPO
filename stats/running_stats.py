from stats_analytics import logistic_regression, interactions_logistic_regression, roc_auc, bootstrap_success_rate, roc_auc_for_interactions, bootstrap_odds_ratios, linear_regression, interactions_odds_ratios, bootstrap_interactions_odds_ratios
import glob
import os
import pandas as pd

files = glob.glob('policy_eval_results/agent_*/EpisodeData_eval_sweep_*.csv')

all_data = []

for file in files:
    dataa = pd.read_csv(file)

    agent_id = os.path.basename(file).replace('EpisodeData_eval_sweep_', '').replace('.csv', '')
    dataa['agent_id'] = int(agent_id)

    all_data.append(dataa)

dataaa = pd.concat(all_data, ignore_index=True)

conditions = pd.read_csv('training_results/manifest.csv').rename(columns = {'index': 'agent_id'})

data = pd.merge(dataaa, conditions, on='agent_id', how='left')

result = logistic_regression(data)
result_linear = linear_regression(data, 'physics_energy_J')
result_linear_exploration = linear_regression(data, 'path_efficiency')

odds_ratios_confidence_intervals = bootstrap_odds_ratios(data)

print("Regular Logistic Regression Results:")
print(result.summary())
print("\nRegular Logistic Regression AIC:", result.aic)

print("Linear regression results:")
print(result_linear.summary())
print(result_linear_exploration.summary())

roc_auc(result, data)
#extra_stats(result)

for variable, interval in odds_ratios_confidence_intervals.items():
    print(f"\n{variable} Bootstrap 95% Confidence Interval: ({interval['lower_percentitle']:.3f}, {interval['upper_percentitle']:.3f})")

overall_success_rate, lower_percentile, upper_percentile = bootstrap_success_rate(data)
print(f"\nOverall Success Rate: {overall_success_rate:.3%}")
print(f"\n95% Bootstrap Confidence Interval: ({lower_percentile:.3%}, {upper_percentile:.3%})")

interactions_result = interactions_logistic_regression(data)
print("\nLogistic Regression with Interactions Results:")

print(interactions_result.summary())
print("\nInteractions Logistic Regression AIC:", interactions_result.aic)

roc_auc_for_interactions(interactions_result, data)

interactions_odds_ratio = interactions_odds_ratios(interactions_result)

print("\nInteractions Odds Ratios:")
for variable, odds_ratio in interactions_odds_ratio.items():
    print(f"\n{variable}: {odds_ratio:.3f}")

bootstrap_interactions_ci = bootstrap_interactions_odds_ratios(data)

for variable, interval in bootstrap_interactions_ci.items():
    print(f"\n{variable} Bootstrap 95% Confidence Interval: ({interval['lower_percentile']:.5f}, {interval['upper_percentile']:.5f})")