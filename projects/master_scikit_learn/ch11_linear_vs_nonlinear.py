""" This file contains code to explore linear versus non-linear models in the workflow """
import pandas as pd
from joblib.testing import param
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder

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
rf = RandomForestClassifier(random_state=1, n_jobs=-1)

imp_constant = SimpleImputer(strategy='constant', fill_value='missing')
imp_ohe = make_pipeline(imp_constant, ohe)

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch'])
)

rf_pipe = make_pipeline(ct, rf)
print(rf_pipe)

print(cross_val_score(rf_pipe, X, y, cv=5, scoring='accuracy').mean())

# parameters for GridSearchCV (as a dictionary)
# - Key: step parameter
# - Value: list of values to try

params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]
params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']
params['columntransformer__countvectorizer__ngram_range'] = [(1,1), (1,2)]
params['columntransformer__simpleimputer__add_indicator'] = [False, True]

# remove the params associated with Logistic Regression model
del params['logisticregression__penalty']
del params['logisticregression__C']
# OR
params = {k:v for k, v in params.items() if k.startswith('column')}

# two-step approach to hyerparameter tuning
# 1. randomized search - test a variety of parameters and values then examine the results for trends
# 2. grid search - use an optimized set of parameters and values base on what is learned from step 1
params['randomforestclassifier__n_estimators'] = [100, 300, 500, 700]
params['randomforestclassifier__min_samples_leaf'] = [1, 2, 3]
params['randomforestclassifier__max_features'] = ['sqrt', None]
params['randomforestclassifier__bootstrap'] = [True, False]

rand_grid = RandomizedSearchCV(rf_pipe, params, cv=5, scoring='accuracy', n_iter=100, random_state=1, n_jobs=-1)
rand_grid.fit(X, y)

results = pd.DataFrame(rand_grid.cv_results_)
print(results.sort_values('rank_test_score'))

# Starting with n_estimators, we see that higher numbers are performing better, which is typical for n_estimators. It seems unlikely that 100 will produce the best result, so we'll exclude that value from our grid search. And since the current best result is at 700, it seems useful to add a value of 900 to our grid search, in case increasing it further is even better.
#
# Do keep in mind that increasing n_estimators also increases the time needed to train the model. You could consider just setting a single large value for n_estimators rather than searching through multiple values, since larger values will generally produce better results up to a certain point, but I prefer to tune this value when computational resources allow for it.
#
# The next parameter to examine is min_samples_leaf. Similar to n_estimators, the lowest value of 1 seems unlikely to produce the best result, so we'll remove it. The current best result is 3, so we'll also try the values 4 and 5 in the grid search.
#
# For max_features, it's clear that None is performing better, so we're no longer going to try sqrt.
#
# For bootstrap, it's clear that True is performing better, so we're no longer going to try False.
#
# And finally, there aren't any clear trends for the transformer parameters, so we'll leave those as-is.
params['randomforestclassifier__n_estimators'] = [100, 300, 500, 700]
params['randomforestclassifier__min_samples_leaf'] = [2, 3, 4, 5]
params['randomforestclassifier__max_features'] = [None]
params['randomforestclassifier__bootstrap'] = [True]

grid = GridSearchCV(rf_pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)

print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

print(grid.predict(X_new))

# tuning two models with a single grid search
logreg = LogisticRegression(solver='liblinear', random_state=1)

both_pipe = Pipeline([('preprocessor', ct), ('classifier', logreg)])

params1 = {}
params1['preprocessor__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
params1['logisticregression__penalty'] = ['l1', 'l2']
params1['logisticregression__C'] = [0.1, 1, 10]
params1['classifier'] = [logreg]

params2 = {}
params2['preprocessor__countvectorizer__ngram_range'] = [(1, 1), (1, 2)]
params2['classifier__n_estimators'] = [300, 500]
params2['classifier__min_samples_leaf'] = [3, 4]
params2['classifier'] = [rf]

both_params = [params1, params2]

both_grid = GridSearchCV(both_pipe, both_params, cv=5, scoring='accuracy', n_jobs=-1)
both_grid.fit(X, y)

print(both_grid.best_score_)
print(both_grid.best_params_)
print(both_grid.best_estimator_)

print(both_grid.predict(X_new))

# Extension to the approach
# First, since the two models have separate parameter dictionaries, you could theoretically tune different preprocessing
# parameters for each model. For example, you could tune different CountVectorizer parameters for logistic regression and
# random forests.
#
# Taking it one step further, you could actually create two different preprocessor objects and tune them using the same
# grid search, just like we tuned two different models using the same grid search. That would allow you, for example,
# to use different encoders when preparing data for your logistic regression and random forest models.

# Tuning two models with a single randomized search
both_rand = RandomizedSearchCV(both_pipe, both_params, cv=5, scoring='accuracy', n_iter=10, random_state=1, n_jobs=-1)
both_rand.fit(X, y)

results = pd.DataFrame(both_rand.cv_results_)
print(results.sort_values('rank_test_score'))
