""" This file contains a review of the ML workflow activities """

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

X = df[['Parch', 'Fare']]
print(X)

y = df[['Survived']] # target as a dataframe with one column

y = df['Survived']  # target as a series

#
# Multilabel vs multiclass
# multilabel - each sample can have more than one label (2-dimensional y (Dataframe))
# multiclass - each sample can have one label (1-dimensional y (Series))

# Building and evaluating a model
logreg = LogisticRegression(solver='liblinear', random_state=1)

# print the model parameters
print(logreg.get_params())

print(cross_val_score(logreg, X, y, cv=3, scoring='accuracy').mean())

# Using the model to make predictions
logreg.fit(X, y)

df_new = pd.read_csv('../../data/titanic_new.csv', nrows=10)
X_new = df_new[['Parch', 'Fare']]

print(logreg.predict(X_new))

# add the predictions to a dataframe
predictions = pd.Series(logreg.predict(X_new), index=X_new.index, name='Prediction')
print(pd.concat([X_new, predictions], axis='columns'))

# determine the confidence level of each prediction
print(logreg.predict_proba(X_new)) # one row for each sample, one column for each class (0,1) in this case

# extract just the predicted probabilities for class 1 by slicing the probabilities array
print(logreg.predict_proba(X_new)[:, 1])



