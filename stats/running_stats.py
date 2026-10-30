from stats.stats_analytics import logistic_regression, interactions_logistic_regression, extra_stats, roc_auc, bootstrap_sampling
import pandas as pd

data = pd.read_csv('metadataabit.csv')

result = logistic_regression(data)
print("Regular Logistic Regression Results:")
print(result.summary())
print("\nRegular Logistic Regression AIC:", result.aic)

overall_success_rate, lower_percentile, upper_percentile = bootstrap_sampling(data['final_success_binary'])
print(f"\nOverall Success Rate: {overall_success_rate:.3%}")
print(f"\n95% Bootstrap Confidence Interval: ({lower_percentile:.3%}, {upper_percentile:.3%})")

roc_auc(result, data)

interactions_result = interactions_logistic_regression(data)
print("\nLogistic Regression with Interactions Results:")
print(interactions_result.summary())
print("\nInteractions Logistic Regression AIC:", interactions_result.aic)