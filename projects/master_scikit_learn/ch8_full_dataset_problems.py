""" This file contains the code for using the full dataset """
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Loading training data
df = pd.read_csv('../../data/titanic_train.csv')
print(df.shape)

# Loading the testing data
df_new = pd.read_csv('../../data/titanic_new.csv')
print(df_new.shape)

print(df.isna().sum())
print(df_new.isna().sum())

# Fix the missing values in a categorical feature
cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

X = df[cols]
y = df['Survived']

ohe = OneHotEncoder()
vect = CountVectorizer()
imp = SimpleImputer()

imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
imp_ohe = make_pipeline(imp_constant, ohe)
imp_ohe.fit_transform(X[['Embarked']])

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age']),
    ('passthrough', ['Parch', 'Fare'])
)

ct.fit_transform(X)
