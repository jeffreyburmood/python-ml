""" This file contains a review of the ML workflow activities """

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

X = df[['Parch', 'Fare']]
print(X)

y = df['Survived']
print(y)

# Building and evaluating a model
logreg = LogisticRegression(solver='liblinear', random_state=1)

print(cross_val_score(logreg, X, y, cv=3, scoring='accuracy').mean())

# Using the model to make predictions
logreg.fit(X, y)

df_new = pd.read_csv('../../data/titanic_new.csv', nrows=10)
X_new = df_new[['Parch', 'Fare']]

print(logreg.predict(X_new))

