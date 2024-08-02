""" This file contains a review of the ML workflow activities """

import pandas as pd
from sklearn.linear_model imp
# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

X = df[['Parch', 'Fare']]
print(X)

y = df['Survived']
print(y)

# Building and evaluating a model
