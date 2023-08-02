""" This file contains examples related to K-fold cross-validation to
    estimate the likely performance of a model.
     
    Better than the train/test split accuracy method. """

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data directly
iris = load_iris()

X = iris.data
y = iris.target

# split up the train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# check the classification accuracy
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"train/test split accuracy = {metrics.accuracy_score(y_test, y_pred)}")

# Now looking at K-fold cross-validation which essentially performs train/test split predictions K times
# and averages out the results

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

# Cross validation example: parameter tuning
# Goal: Select the best tuning parameters (hyperparameters) for KNN on the iris dataset

from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)
# 10-fold cross validation
scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")

print("Scores results: \n")
print(scores)
# use average accuracy as an eestimate of out-of-sample accuracy
print("Scores Mean: \n")
print(scores.mean())

# find an optimal value for K for KNN
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    k_scores.append(scores.mean())

print(k_scores)

import matplotlib.pyplot as plt

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-validated Accuracy")
# plt.show()

# Generally, best to select the simplist model, for KNN, higher K produces simpler model so K=20 is best


# Cross validation example: model selection
# Compare KNN vs Logistic Regression for Iris

knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring="accuracy").mean())

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring="accuracy").mean())

# Cross-validation example: feature selection
# Goal: should Newpapers feature be included in the advertising dataset

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv(
    "/Users/jeffrey/SynologyDrive/projects/Python-Programming/Training/python-ml/data/Advertising.csv",
    index_col=0,
)

feature_cols = ["TV", "Radio", "Newspaper"]
X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring="neg_mean_squared_error")
print(scores)
# print(metrics.get_scorer_names())

mse_scores = -scores
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores.mean())

feature_cols = ["TV", "Radio"]
X = data[feature_cols]
print(
    np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring="neg_mean_squared_error")).mean()
)

# searching for optimal tuning parameters
# goal: more efficient parameter tuning using GridSearchCV

from sklearn.model_selection import GridSearchCV

X = iris.data
y = iris.target

k_range = range(1, 31)
# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy", n_jobs=-1)

# fit the grid with data
grid.fit(X, y)

# print(f"Grid CV results:\n {grid.cv_results_}")

# examine the first tuple
print(grid.cv_results_["params"][0])
print(grid.cv_results_["split0_test_score"])
print(grid.cv_results_["mean_test_score"][0])

grid_mean_scores = [result for result in grid.cv_results_["mean_test_score"]]

plt.plot(k_range, grid_mean_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross Validated Accuracy")
# plt.show()

# examine the best model
print("Best Values")
print(f"Best Score {grid.best_score_}")
print(f"Best Params {grid.best_params_}")
print(f"Best Estimator {grid.best_estimator_}")
