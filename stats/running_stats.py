<<<<<<< HEAD
from stats_analytics import logistic_regression, interactions_logistic_regression, extra_stats, roc_auc, bootstrap_success_rate, roc_auc_for_interactions, bootstrap_odds_ratios, linear_regression
=======
from stats_analytics import logistic_regression, interactions_logistic_regression, extra_stats, roc_auc, bootstrap_success_rate, roc_auc_for_interactions, bootstrap_odds_ratios
>>>>>>> f59ee730d63d1d81864b197326cc29ca4d16c0e8
import pandas as pd
import glob
import os

files = glob.glob('policy_eval_results/*/EpisodeData_eval_sweep_*.csv')

all_data = []

for file in files:
    dataa = pd.read_csv(file)

    agent_id = os.path.basename(file).replace('EpisodeData_eval_sweep_', '').replace('.csv', '')
    dataa['agent_id'] = agent_id

    all_data.append(dataa)

dataaa = pd.concat(all_data, ignore_index=True)

conditions = pd.read_csv('training_results/manifest.csv')

data = pd.merge(dataaa, conditions, on='agent_id', how='left')

result = logistic_regression(data)
result_linear = linear_regression(data)
odds_ratios_confidence_intervals = bootstrap_odds_ratios(data)

print("Regular Logistic Regression Results:")
print(result.summary())
print("\nRegular Logistic Regression AIC:", result.aic)
roc_auc(result, data)
extra_stats(result, data)
for variable, interval in odds_ratios_confidence_intervals.items():
    print(f"\n{variable} Bootstrap 95% Confidence Interval: ({interval['lower_percentitle']:.3f}, {interval['upper_percentitle']:.3f})")

<<<<<<< HEAD
print("Linear regression results:")
print(result_linear.summary())

roc_auc(result, data)
extra_stats(result, data)

for variable, interval in odds_ratios_confidence_intervals.items():
    print(f"\n{variable} Bootstrap 95% Confidence Interval: ({interval['lower_percentitle']:.3f}, {interval['upper_percentitle']:.3f})")

=======
>>>>>>> f59ee730d63d1d81864b197326cc29ca4d16c0e8
overall_success_rate, lower_percentile, upper_percentile = bootstrap_success_rate(data)
print(f"\nOverall Success Rate: {overall_success_rate:.3%}")
print(f"\n95% Bootstrap Confidence Interval: ({lower_percentile:.3%}, {upper_percentile:.3%})")

<<<<<<< HEAD
=======
interactions_result = interactions_logistic_regression(data)
>>>>>>> f59ee730d63d1d81864b197326cc29ca4d16c0e8
print("\nLogistic Regression with Interactions Results:")
interactions_result = interactions_logistic_regression(data)

print(interactions_result.summary())
print("\nInteractions Logistic Regression AIC:", interactions_result.aic)
<<<<<<< HEAD

=======
>>>>>>> f59ee730d63d1d81864b197326cc29ca4d16c0e8
roc_auc_for_interactions(interactions_result, data)