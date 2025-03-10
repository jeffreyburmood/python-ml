""" this file contains code for performing feature selection for models
    Automated methods for feature selection
    - Intrinsic methods
        - feature selection happens automatically during model building
    - Filter methods
    - Wrapper methods """
import numpy as np
# Intrinsic methods: L1 regularization
# For Logistic Regression the tuning parameters are:
#   - penalty - type of regularization
#   - C - amount of regularization
# vc_params['votingclassifier__clf1__penalty'] = ['l1', 'l2']
# vc_params['votingclassifier__clf1__C'] = [1, 10]
# Regularization shrinks model coefficient to help the model to generalize
# L1 regularization shrinks some coefficients to zeros which removes those features

# Filter methods: Statistical test-based scoring
#   - each feature is scored by it's relationship to the target
#   - top scoring features (most informative) are provided to the model
#
# Filter methods: Model-based scoring
#   - Scores each feature using a specified model
#       - model is fit on all features
#       - coefficients or feature importances are used as scores
#   - Passes to the prediction model features that score above a specified threshold
#
# Models that can be used:
#   - Logistic regression
#   - Linear SVC
#   - Tree-based
#   - Any other model with coefficients or feature importances
#
# The main advantage is that filter methods tend to run very quickly, though it's worth noting that some statistical
# tests used with SelectPercentile and ensemble methods used with SelectFromModel can run quite slowly.
#
# The main disadvantage is that there's a disconnect between how features are being scored and their predictive value.
# In other words, the chi2 scores or coefficient values or feature importance scores are not a perfect measure of whether a particular feature will help a model make more accurate predictions. Thus, it's entirely possible for informative features to receive low scores and be removed from a model, and for uninformative features to receive high scores and be kept in a model. One particular case of note is that the feature importance scores generated by tree-based models will be artificially low for any features which are highly correlated, which may result in important features being removed.
#
# The other disadvantage of filter methods is that scores are calculated only once. This ignores the fact that as you
# remove certain features, the importance of other features may change.

# Wrapper methods: Recursive feature elimination (RFE)
#
#   Wrapper methods vs filter methods
#   - filter methods - features are scored once
#   - wrapper methods - features are scored multiple times
#
# SelectFromModel vs RFE
#   - SelecrFromModel scores features a single time
#   - RFE scores features many times - may better capture relationships between features

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE

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

ct = make_column_transformer(
    (imp_ohe, ['Embarked', 'Sex']),
    (vect, 'Name'),
    (imp, ['Age', 'Fare']),
    ('passthrough', ['Parch'])
)

pipe = make_pipeline(ct, logreg)

print(cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean())

# Filter method - SelectPercentile
#   - scored each feature using specified statistical test
#   - passes to the model the specified percentage of features
#
# SelectPercentile vs SelectKBest
# SelectPercentile - specify percentage of features to keep
# SelectKBest - specify the number of features to keep

selection = SelectPercentile(chi2, percentile=50)

fs_pipe = make_pipeline(ct, selection, logreg)

print(cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean())

# Model-based scoring - Logistic Regression
logreg_selection = LogisticRegression(solver='liblinear', penalty='l1', random_state=1)

selection = SelectFromModel(logreg_selection, threshold='mean')

fs_pipe = make_pipeline(ct, selection, logreg)

print(cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean())

# Model-based scoring - Random Forest
et_selection = ExtraTreesClassifier(n_estimators=100, random_state=1)

selection = SelectFromModel(et_selection, threshold='mean')

fs_pipe = make_pipeline(ct, selection, logreg)

print(cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean())

# Tune the feature selection parameters, transformer parameters, and models parameters all at the same time
params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]
params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']
params['columntransformer__countvectorizer__ngram_range'] = [(1,1), (1,2)]
params['columntransformer__simpleimputer__add_indicator'] = [False, True]

fs_params = params.copy()
fs_params['selectfrommodel__threshold'] = ['mean', '1.5*mean', -np.inf]

fs_grid = GridSearchCV(fs_pipe, fs_params, cv=5, scoring='accuracy', n_jobs=-1)
fs_grid.fit(X, y)

print(fs_grid.best_score_)
print(fs_grid.best_params_)

# Wrapper methods - RPE
selection = RFE(logreg_selection, step=10)

fs_pipe = make_pipeline(ct, selection, logreg)

print(cross_val_score(fs_pipe, X, y, cv=5, scoring='accuracy').mean())
params = {}
params['logisticregression__penalty'] = ['l1', 'l2']
params['logisticregression__C'] = [0.1, 1, 10]
params['columntransformer__pipeline__onehotencoder__drop'] = [None, 'first']
params['columntransformer__countvectorizer__ngram_range'] = [(1,1), (1,2)]
params['columntransformer__simpleimputer__add_indicator'] = [False, True]

fs_params = params.copy()
fs_params['rfe__n_features_to_select'] = [None, 500]

fs_grid = GridSearchCV(fs_pipe, fs_params, cv=5, scoring='accuracy', n_jobs=-1)
fs_grid.fit(X, y)

print(fs_grid.best_score_)
print(fs_grid.best_params_)
