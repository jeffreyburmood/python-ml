""" This file contains code for improving workflow with ColumnTransformer and Pipeline
    ColumnTransformer - apply different preprocessing steps to different columns
    Pipeline - apply the same workflow to training data and new data """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

cols = ['Parch', 'Fare', 'Embarked', 'Sex']
X = df[cols]
print(X)

y = df['Survived'] # target as a dataframe with one column

ohe = OneHotEncoder()

from sklearn.compose import make_column_transformer
# drop - drop all other columns
ct = make_column_transformer((ohe, ['Embarked', 'Sex']), remainder='drop')
print(ct.fit_transform(X))

# passthrough - keep and pass through any other columns
ct = make_column_transformer((ohe, ['Embarked', 'Sex']), remainder='passthrough')
print(ct.fit_transform(X))
print(ct.get_feature_names_out())

# revised - can specify columns to 'drop' or 'passthrough'
ct = make_column_transformer((ohe, ['Embarked', 'Sex']), ('passthrough', ['Parch', 'Fare']))
print(ct.fit_transform(X))

# revised - don't need to list columns by name
# ct = make_column_transformer((ohe, ['Embarked', 'Sex']), ('drop', ['Fare']), remainder='passthrough')
# ct = make_column_transformer((ohe, ['Embarked', 'Sex']), ('passthrough', ['Parched'), remainder='drop')

from sklearn.pipeline import make_pipeline
# Building and evaluating a model
logreg = LogisticRegression(solver='liblinear', random_state=1)

pipe = make_pipeline(ct, logreg)
print(pipe)
print(pipe.fit(X, y))

# make predictions using a pipeline
df_new = pd.read_csv('../../data/titanic_new.csv', nrows=10)
X_new = df_new[cols]
print(X_new)
print(pipe.predict(X_new))