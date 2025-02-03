""" This file contains code to explore how to handle missing data """
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

# Ways to handle missing values
# - drop rows with missing data (may discard too much data, may obscure patterns, doesn't help with new data
# - drop columns that have missing data (may discard important features)
# - impute missing values (imputed values may not match the true values)

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']
X = df[cols]
print(X)

print(X.dropna())

print(X.dropna(axis='columns'))

# Simple imputation strategies
# - Mean (default)
# - Median
# - most frequent
# - user-defined
imp = SimpleImputer()

print(imp.fit_transform(X[['Age']]))
print(imp.statistics_)

y = df['Survived'] # target as a dataframe with one column

df_new = pd.read_csv('../../data/titanic_new.csv', nrows=10)

ohe = OneHotEncoder()

logreg = LogisticRegression(solver='liblinear', random_state=1)

# using a text encoder
# OneHotEncoder - each full name is treated as a category (not recommended)
# CountVectorizer - each word in a name is treated independently (recommended)
#   - uses 1-dimensional input (Series)
#   - CountVectorizer converts text into a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
dtm = vect.fit_transform(df['Name'])

ct = make_column_transformer((ohe, ['Embarked', 'Sex']),
                             (vect, 'Name'),
                             (imp, ['Age']),
                             ('passthrough', ['Parch', 'Fare']))
print(ct.fit_transform(X))
print(ct.get_feature_names_out())

pipe = make_pipeline(ct, logreg)
print(pipe.fit(X, y))
print(pipe.named_steps['columntransformer'].named_transformers_['simpleimputer'].statistics_)

X_new = df_new[cols]
print(pipe.predict(X_new))
