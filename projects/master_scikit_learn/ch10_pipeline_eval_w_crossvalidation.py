""" This file contains code for evaluation and tuning a pipeline with cross validation """
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Loading training data
df = pd.read_csv('../../data/titanic_train.csv')

# Loading the testing data
df_new = pd.read_csv('../../data/titanic_new.csv')

# Fix the missing values in a categorical feature
cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name', 'Age']

X = df[cols]
y = df['Survived']

ohe = OneHotEncoder()
vect = CountVectorizer()
imp = SimpleImputer()
logreg = LogisticRegression(solver='liblinear', random_state=1)

imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
imp_ohe = make_pipeline(imp_constant, ohe)

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch'])
)

pipe = make_pipeline(ct, logreg)
pipe.fit(X, y)

print(cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean())

# parameters for GridSearchCV (as a dictionary)
# - Key: step parameter
# - Value: list of values to try

params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

results = pd.DataFrame(grid.cv_results_)
print(results.sort_values('rank_test_score'))

# Tuning the transformers
print(pipe.named_steps['columntransformer'].named_transformers_)

# show all pipeline parameters
print(list(pipe.get_params().keys()))

params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']
params['columntransformer__countvectorizer__ngram_range'] = [(1,1), (1,2)]
params['columntransformer__simpleimputer__add_indicator'] = [False, True]

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

results = pd.DataFrame(grid.cv_results_)
print(results.sort_values('rank_test_score'))

print(grid.best_score_)
print(grid.best_params_)
