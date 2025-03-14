""" This file contains code related to standardizing model features.

    Numerical feature standardization
        - some models assume that features are centered around zero and have similar variances

    Feature standardization tends to be useful any time a model considers the distance between features,
    such as K-Nearest Neighbors and Support Vector Machines.

    Feature standardization also tends to be useful for any models that incorporate regularization,
    such as a linear regression or logistic regression model with an L1 or L2 penalty, though we saw earlier in the chapter that this doesn't apply to all solvers.

    Notably, feature standardization will not benefit any tree-based models such as random forests. """

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

# Loading training data
df = pd.read_csv('../../data/titanic_train.csv')

# Loading the testing data
df_new = pd.read_csv('../../data/titanic_new.csv')

# Fix the missing values in a categorical feature
cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

X = df[cols]
y = df['Survived']
X_new = df_new[cols]

ohe = OneHotEncoder()
vect = CountVectorizer()
imp = SimpleImputer()
logreg = LogisticRegression(solver='liblinear', random_state=1)

imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
imp_ohe = make_pipeline(imp_constant, ohe)

# standardizing just numerical features
#
scaler = StandardScaler()

imp_scaler = make_pipeline(imp, scaler)

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp_scaler, ['Age', 'Fare', 'Parch'])
)

scaler_pipe = make_pipeline(ct, logreg)

print(cross_val_score(scaler_pipe, X, y, cv=5, scoring='accuracy').mean())

# standardizing all features
# can't use StandardScaler because ColumnTransformer outputs a sparse matrix and StandardScaler would create a dense matrix
#
scaler = MaxAbsScaler()

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch'])
)

# add the scaler as a step in the pipeline
scaler_pipe = make_pipeline(ct, scaler, logreg)

print(cross_val_score(scaler_pipe, X, y, cv=5, scoring='accuracy').mean())

# to see the scaling that was applied
# last three params are the max values of Age, Fare, and Parch that will be applied to X_new when making predictions
scaler_pipe.fit(X, y)
print(scaler_pipe.named_steps['maxabsscaler'].scale_)

# using GridSearch to determine whether to use feature standardization
scaler_params = {}
scaler_params['logisticregression__C'] = [0.1, 1, 10]
scaler_params['maxabsscaler'] = ['passthrough', MaxAbsScaler()]

scaler_grid = GridSearchCV(scaler_pipe, scaler_params, cv=5, scoring='accuracy', n_jobs=-1)
scaler_grid.fit(X, y)

# results show best approach is to use feature standardization rather than just passthrough
print(scaler_grid.best_params_)