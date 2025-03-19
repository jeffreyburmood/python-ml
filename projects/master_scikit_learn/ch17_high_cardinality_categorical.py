""" There are two types of categorical features that we've covered in the course:

    A nominal feature has categories that are unordered, such as Embarked and Sex.
    An ordinal feature has categories with an inherent logical ordering, such as Pclass.

    For a nominal feature, you should use OneHotEncoder, and it will output one column for each category.
    For an ordinal feature that is already encoded numerically, you should leave it as-is.
    And for an ordinal feature that is encoded as strings, you should use OrdinalEncoder, and it will output a
        single column using the category ordering that you define.

    we're going to explore when you have high-cardinality categorical features, which are categorical features
    with lots of unique values"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from projects.master_scikit_learn.ml_workflow_review import logreg

# Loading and exploring data
census = pd.read_csv('../../data/census.csv')

# categorical features in the census dataset
# - high-cardinality (3 of 8): education, occupation, native-country
# - nominal (7 of 8): all except education (which is ordinal ut will be treated as nominal for this experiment

print(census['class'].value_counts(normalize=True))

census_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

census_X = census[census_cols]
census_y = census['class']

ohe = OneHotEncoder(handle_unknown='ignore')

cats = [census_X[col].unique() for col in census_X[census_cols]]

oe = OrdinalEncoder(categories=cats)

ohe.fit_transform(census_X)
oe.fit_transform(census_X)

ohe_logreg = make_pipeline((ohe, logreg))
oe_logreg = make_pipeline(oe, logreg)

print(cross_val_score(ohe_logreg, census_X, census_y, cv=5, scoring='accuracy').mean())
print(cross_val_score(oe_logreg, census_X, census_y, cv=5, scoring='accuracy').mean())

# If you have nominal features, and you're using a linear model, you should definitely use OneHotEncoder, regardless of
# whether the features have high cardinality.
#
# If you have nominal features, and you're using a non-linear model, you can try using OneHotEncoder, and you can try
# using OrdinalEncoder without defining the category ordering, and then see which option performs better.
#
# If you have ordinal features, regardless of the type of model, you can try using OneHotEncoder, and you can try
# using OrdinalEncoder while defining the category ordering, and then see which option performs better.