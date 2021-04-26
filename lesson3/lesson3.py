import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Task 1
boston = load_boston()
data = boston["data"]
feature_names = boston["feature_names"]
target = boston["target"]
X = pd.DataFrame(data, columns=feature_names)
y = pd.DataFrame(target, columns=["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten()
})

print(check_test.head(10))

r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

# Task 2
model = RandomForestRegressor(max_depth=12, n_estimators=1000, random_state=42)
model.fit(X_train, y_train.values[:, 0])
y_pred = model.predict(X_test)

r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

# Модель на основе деревьев работает точнее

# Task 3
help(RandomForestRegressor.feature_importances_)
importances = model.feature_importances_
print("Sum of features: " + str(sum(importances)))

feats = {}
for feature, importance in zip(feature_names, model.feature_importances_):
    feats[feature] = importance

sorted_tuples = sorted(feats.items(), key=lambda item: item[1], reverse=True)
print(f"The most important features: {sorted_tuples[0]}, {sorted_tuples[1]}")

