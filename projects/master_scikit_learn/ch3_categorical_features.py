""" This file represents Chapter 3: Encoding categorical features """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

X = df[['Parch', 'Fare']]
print(X)

# sparse - more efficient and performant
# dense - more readable
ohe = OneHotEncoder(sparse_output=False)

# single brackets - outputs as Series, could be interpreted as a single feature
# double brackets - outputs as a single column feature
# results in one column for each unique value of the feature
# fit: learn the categories
# transform: create the feature matrix using those categories
# CORRECT PROCESS:
# - run fit_transform() on training data
# - run only transform() on live (testing) data
ohe.fit_transform(df[['Embarked', 'Sex']])
print(ohe.categories_)

# encoding for ordinal categories uses a different encoder
from sklearn.preprocessing import OrdinalEncoder
df_ordinal = pd.DataFrame({'Class': ['third', 'first', 'second', 'third'], 'Size': ['S', 'S', 'L', 'XL']})
oe = OrdinalEncoder(categories=[['first', 'second', 'third'], ['S', 'M', 'L', 'XL']])
print(oe.fit_transform(df_ordinal))

# encoding numeric features as ordinal features
print(df['Fare'])
from sklearn.preprocessing import KBinsDiscretizer
kb = KBinsDiscretizer(n_bins=3, strategy='quantile', encode='ordinal')
print(kb.fit_transform(df[['Fare']]))


y = df[['Survived']] # target as a dataframe with one column

y = df['Survived']  # target as a series
