""" This file contains code related to feature engineering.

    Can create some custom features for your model. This is usually because you believe
    that your model could learn more from a particular feature if the feature was represented
    in a different way or combined with another feature.

    Often, feature engineering is done using pandas on the original dataset, and then the updated
    dataset is passed to scikit-learn. However, you can actually do feature engineering within scikit-learn
    using custom transformers. """
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.preprocessing import FunctionTransformer

# Loading training data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

print(np.ceil(df[['Fare']]))

# In order to do this transformation within scikit-learn, we need to convert the ceil function into a scikit-learn
# transformer using the FunctionTransformer class.
#
# We simply pass the numpy ceil() function to FunctionTransformer
# FunctionTransformer should only be used with stateless transformations. A transformation is considered stateless
# if it doesn't learn any information during the fit step.
ceiling = FunctionTransformer(np.ceil)

print(ceiling.fit_transform(df[['Fare']]))

# Like any transformer, ceiling can be included in a ColumnTransformer

ct = make_column_transformer(
    (ceiling, ['Fare'])
)

# ColumnTransformer always outputs a NumPy array or a sparse matrix, and in this case it outputs a NumPy array
print(ct.fit_transform(df))

# update Age feature to round all ages below 5 to 5, and al ages above 60 to 60.
# the clip() function has two required parameters
print(np.clip(df[['Age']], a_min=5, a_max=60))

clip = FunctionTransformer(np.clip, kw_args={'a_min':5, 'a_max':60})

ct = make_column_transformer(
    (ceiling, ['Fare']),
    (clip, ['Age'])
)

print(ct.fit_transform(df))

# next, use the first letter of Cabin to find the deck the passenger was on
# build a custom function to extract the first letter and can handle a dataframe of numpy array
#
# notice that the output is a 1D object, namely a pandas Series. This is problematic because a function
# (once transformed) must return 2D output in order to be used in a ColumnTransformer.
#
# To resolve this, we'll use the DataFrame apply method with a lambda. It still extracts the first character,
# but notice that it now returns 2D output. Also notice that it accepts 2D input, which means that it will be able to
# operate on multiple columns

def first_letter(dataf):
    return pd.DataFrame(dataf).apply(lambda x: x.str.slice(0, 1))

print(first_letter(df[['Cabin']]))

letter = FunctionTransformer(first_letter)

ct = make_column_transformer(
    (ceiling, ['Fare']),
    (clip, ['Age']),
    (letter, ['Cabin'])
)

print(ct.fit_transform(df))

# next, suppose we want a feature representing number of members in the party ['SibSp', 'Parch']
print(df[['SibSp', 'Parch']].sum(axis=1))

# but this outputs a 1D object which cannot be used with the Column Transformer so convert to a numpy array and reshape
print(np.array(df[['SibSp', 'Parch']]).sum(axis=1).reshape(-1, 1))

def sum_cols(df):
    return np.array(df).sum(axis=1).reshape(-1, 1)

total = FunctionTransformer(sum_cols)
total.fit_transform(df[['SibSp', 'Parch']])

ct = make_column_transformer(
    (ceiling, ['Fare']),
    (clip, ['Age']),
    (letter, ['Cabin']),
    (total, ['SibSp', 'Parch'])
)

print(ct.fit_transform(df))

# revising the transformers
# issues to handle
#   - Cabin and SibSp weren't originally included
#   - Fare and Age have missing values
#   - Cabin is non-numeric and has missing values

# Loading training data
df = pd.read_csv('../../data/titanic_train.csv')

# Loading the testing data
df_new = pd.read_csv('../../data/titanic_new.csv')

# Fix the missing values in a categorical feature
cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age', 'Cabin', 'SibSp']

X = df[cols]
y = df['Survived']
X_new = df_new[cols]

ohe = OneHotEncoder()
vect = CountVectorizer()
imp = SimpleImputer()
logreg = LogisticRegression(solver='liblinear', random_state=1)

imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
imp_ohe = make_pipeline(imp_constant, ohe)
imp_ceiling = make_pipeline(imp, ceiling)
imp_clip = make_pipeline(imp, clip)

print(X['Cabin'].str.slice(0,1).value_counts(dropna=False))

ohe_ignore = OneHotEncoder(handle_unknown='ignore')
letter_imp_ohe = make_pipeline(letter, imp_constant, ohe_ignore)

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp_ceiling, ['Fare']),
    (imp_clip, ['Age']),
    (letter_imp_ohe, ['Cabin']),
    (total, ['SibSp', 'Parch'])
)

ct.fit_transform(X)

pipe = make_pipeline(ct, logreg)

print(cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean())

# dealing with feature data type issues
#   - numbers are represented as strings in the dataset
demo = pd.DataFrame({'A': ['10', '20', '30'],
                     'B': ['40', '50', '60'],
                     'C': [70, 80, 90],
                     'D': ['x', 'y', 'z']})
print(demo.dtypes)

# introduce an empty string that will need to be handled
# A strategy is to use the pandas function to_numeric() and apply it to each of the columns.
# to_numeric() replaced the empty string with NaN. Column A is now an integer column, and
# column B is now a float column since integer columns don't currently support NumPy's NaN value.

demo.loc[2, 'B'] = ''
print(demo)

print(demo[['A', 'B']].apply(pd.to_numeric).dtypes)

def make_number(df):
    return pd.DataFrame(df).apply(pd.to_numeric)

number = FunctionTransformer(make_number)
print(number.fit_transform(demo[['A', 'B']]))

# handling Date features
df = pd.read_csv('../../data/ufo.csv', nrows=10)

print(df.dtypes)

# convert each column entry to datetime format and then extract the day from each row element datetime object
def day_of_month(df):
    return pd.DataFrame(df).apply(lambda x: pd.to_datetime(x).dt.day)

day = FunctionTransformer(day_of_month)
print(day.fit_transform(df[['Date']]))