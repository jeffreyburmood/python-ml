""" This file contains code related to ensembling , or combining, multiple models that is more accurate than a single model.
    - Regression - average the predictions
    - Classification - average the predicted probabilities or let the classifiers vote on the class """

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
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
X_new = df_new[cols]

ohe = OneHotEncoder()
vect = CountVectorizer()
imp = SimpleImputer()
rf = RandomForestClassifier(random_state=1, n_jobs=-1)
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
rf_pipe = make_pipeline(ct, rf)

# Voting options
# soft - average the predicted probabilities
# hard - majority vote using class predictions
# Choosing a voting strategy
# soft voting:
# - preferred if there's an even number of models (especially two)
# - preferred if all models are well-calibrated
# - only works if all models have the predict_proba method
# hard voting:
# - preferred if some models are not well calibrated
# - does ot require the predict_proba method

vc = VotingClassifier([('clf1', logreg), ('clf2', rf)], voting='soft', n_jobs=-1)

vc_pipe = make_pipeline(ct, vc)

print(cross_val_score(vc_pipe, X, y, cv=5, scoring='accuracy').mean())

# Now use hard voting
# in case of a tie, hard voting always chooses class 0 which may result i a misleading result
vc2 = VotingClassifier([('clf1', logreg), ('clf2', rf)], voting='hard', n_jobs=-1)

vc_pipe2 = make_pipeline(ct, vc2)

print(cross_val_score(vc_pipe2, X, y, cv=5, scoring='accuracy').mean())

params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]
params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']
params['columntransformer__countvectorizer__ngram_range'] = [(1,1), (1,2)]
params['columntransformer__simpleimputer__add_indicator'] = [False, True]

vc_params = {k:v for k, v in params.items() if k.startswith('column')}
vc_params['votingclassifier__clf1__penalty'] = ['l1', 'l2']
vc_params['votingclassifier__clf1__C'] = [1, 10]
vc_params['votingclassifier__clf2__n_estimators'] = [100, 300]
vc_params['votingclassifier__clf2__min_samples_leaf'] = [2, 3]

vc_grid = GridSearchCV(vc_pipe, vc_params, cv=5, scoring='accuracy', n_jobs=-1)
vc_grid.fit(X, y)

print(vc_grid.best_score_)
print(vc_grid.best_params_)

