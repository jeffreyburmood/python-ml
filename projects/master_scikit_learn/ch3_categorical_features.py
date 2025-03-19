""" This file represents Chapter 3: Encoding categorical features

    There are two types of categorical features:

    A nominal feature has categories that are unordered, such as Embarked and Sex.
    An ordinal feature has categories with an inherent logical ordering, such as Pclass.
    So far, here's the advice for encoding nominal and ordinal features:

    For a nominal feature, you should use OneHotEncoder, and it will output one column for each category.
    For an ordinal feature that is already encoded numerically, you should leave it as-is.
    And for an ordinal feature that is encoded as strings, you should use OrdinalEncoder, and it will output a
        single column using the category ordering that you define."""

import pandas as pd
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
