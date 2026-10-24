from stats_analytics import logistic_regression, interactions_logistic_regression, extra_stats, roc_auc
import pandas as pd

data = pd.read_csv('metadataabit.csv')

result = logistic_regression(data)
print("Regular Logistic Regression Results:")
print(result.summary())
print("\nRegular Logistic Regression AIC:", result.aic)

extra_stats(result, data)

roc_auc(result, data)

interactions_result = interactions_logistic_regression(data)
print("\nLogistic Regression with Interactions Results:")
print(interactions_result.summary())
print("\nInteractions Logistic Regression AIC:", interactions_result.aic)