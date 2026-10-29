from stats_analytics import logistic_regression, interactions_logistic_regression, extra_stats, roc_auc, bootstrap_success_rate, roc_auc_for_interactions, bootstrap_odds_ratios
import pandas as pd

data = pd.read_csv('manifest.csv')

result = logistic_regression(data)
odds_ratios_confidence_intervals = bootstrap_odds_ratios(data)
print("Regular Logistic Regression Results:")
print(result.summary())
print("\nRegular Logistic Regression AIC:", result.aic)
roc_auc(result, data)
extra_stats(result, data)
for variable, interval in odds_ratios_confidence_intervals.items():
    print(f"\n{variable} Bootstrap 95% Confidence Interval: ({interval['lower_percentitle']:.3f}, {interval['upper_percentitle']:.3f})")

overall_success_rate, lower_percentile, upper_percentile = bootstrap_success_rate(data['final_success_binary'])
print(f"\nOverall Success Rate: {overall_success_rate:.3%}")
print(f"\n95% Bootstrap Confidence Interval: ({lower_percentile:.3%}, {upper_percentile:.3%})")

interactions_result = interactions_logistic_regression(data)
print("\nLogistic Regression with Interactions Results:")
print(interactions_result.summary())
print("\nInteractions Logistic Regression AIC:", interactions_result.aic)
roc_auc_for_interactions(interactions_result, data)