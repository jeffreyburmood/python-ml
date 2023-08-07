""" Working with linear regression models using pandas and seaborn 

    Seaborn - a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures.
    
    """

import pandas as pd

data = pd.read_csv(
    "/Users/jeffrey/SynologyDrive/projects/Python-Programming/Training/python-ml/data/Advertising.csv",
    index_col=0,
)

# display first 5 rows
print(data.head())

# display last 5 rows
print(data.tail())

# display the shape of the DataFrame
print(data.shape)

import seaborn as sns
import matplotlib.pyplot as plt

# visualize the relationship between the features and the response using scatterplots
sns.pairplot(
    data,
    x_vars=["TV", "Radio", "Newspaper"],
    y_vars="Sales",
    height=7,
    aspect=0.7,
    kind="reg",
)
# plt.show()

# X will be a pandas Datagrame, and y will be a pandas series

feature_cols = ["TV", "Radio", "Newspaper"]

X = data[feature_cols]
# equivalent
X = data[["TV", "Radio", "Newspaper"]]  # dataframe

print(X.head())
print(type(X))
print(X.shape)

y = data["Sales"]  # series
# equivalent
y = data.Sales

print(y.head())
print(type(y))
print(y.shape)

# split into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# linear regression modeling

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)

print(reg.intercept_)
print(reg.coef_)

# make prediction

y_pred = reg.predict(X_test)

# model evaluation via Mean Absolute Error (MAE)
from sklearn import metrics

print(metrics.mean_absolute_error(y_test, y_pred))

# model evaluation via Mean Squared Error (MSE)
print(metrics.mean_squared_error(y_test, y_pred))

# model evaluation via Root Mean Squared Error (RMSE)
import numpy as np

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# feature selection (remove Newspaper feature)
feature_cols = ["TV", "Radio"]

X = data[["TV", "Radio"]]  # dataframe
y = data["Sales"]  # series

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

reg = LinearRegression()

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(
    np.sqrt(metrics.mean_squared_error(y_test, y_pred))
)  # slightly better result by removing feature!
