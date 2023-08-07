""" Evaluation a Classification Model 
    1. What are some commn evaluation procedures?
    2. What is the usage of classification accuracy?
    3. How does a confusion matrix describe the performance of a classifier?
    4. What metrics can be computed from a confusion matrix?
"""

# Pima Indian Diabetes dataset
import pandas as pd

url = "/Users/jeffrey/SynologyDrive/projects/Python-Programming/Training/python-ml/data/diabetes.csv"
col_names = [
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "label",
]
pima = pd.read_csv(url)

print(pima.head())

# select the features to use and create X and y
feature_cols = ["Pregnancies", "Insulin", "BMI", "Age"]
X = pima[feature_cols]
y = pima.Outcome

# build the training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train a logistic regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make the class predictions for the testing set
y_pred_class = logreg.predict(X_test)

# calculate the accuracy
from sklearn import metrics

print(f"Model Accuracy: {metrics.accuracy_score(y_test, y_pred_class)}")

# null accuracy - predicting the most frequent class (in this case zero)
# examing the class distribution of the testing set
print(y_test.value_counts())

# calculate the percentage of ones
print(y_test.mean())

# calculate the percentage of zeros
print(1 - y_test.mean())

# calcluate null accuracy for binary classification problems coded as 0/1
print(max(y_test.mean(), 1 - y_test.mean()))

# calculate null accuracy for multi-class classification problems
print(y_test.value_counts().head(1) / len(y_test))

# comparing true and predicted response values
# conclusion - classification accuracy does not take into account the underlying distribution of response values
print(f"True: {y_test.values[0:25]}")
print(f"Pred: {y_pred_class[0:25]}")

###
### Confusion Matrix - table that describes the performance of a classification model
###
# IMPORTANT: first argument is true values, second argument is predicted values
confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(f"Confusion Matric: {confusion}")

# Basic terminology
# - True Positives (TP) - correctly predicted had diabetes
# - True Negatives (TN) - correctly predicted did NOT have diabetes
# - False Positives (FP) - incorrectly predicted had diabetes
# - Flase Negatives (FN) - incorrectly predicted did NOT have diabetes

# slice up the confusion matrix for calculations
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# Classification accuracy
print(
    f"classification accuracy from confusion matrix: {(TP + TN) / float(TP + TN + FP + FN)}"
)
print(
    f"Classification accuracy calculated:            {metrics.accuracy_score(y_test, y_pred_class)}"
)

# Classification error
print(
    f"classification error from confusion matrix: {(FP + FN) / float(TP + TN + FP + FN)}"
)
print(
    f"Classification error calculated:            { 1 - metrics.accuracy_score(y_test, y_pred_class)}"
)

# Sensitivity - when the actual value is positive, how often is the rediction correct?
print(f"classification sensitivity from confusion matrix: {TP / float(TP + FN)}")
print(
    f"Classification sensitivty calculated:             {metrics.recall_score(y_test, y_pred_class)}"
)

# Specificity - when the actual value is negative, how often is the prediction correct?
print(f"classification specificity from confusion matrix: {TN / float(TN + FP)}")

# False Positive Rate - when the actual valuve is negative, how often is the prediction correct? (1-specificity)
print(
    f"classification false positive rate from confusion matrix: {FP / float(TN + FP)}"
)

# Precision - when the a positive value is predicted, how often is the prediction correct?
print(f"classification precision from confusion matrix: {TP / float(TP + FP)}")
print(
    f"Classification precision calculated:             {metrics.precision_score(y_test, y_pred_class)}"
)

# analyzing classifier performance
# Area Under the Curve (AUC) is usefule as a single number summary of classifier performance
# If you randomly chose one positive and one negative observation, AUC represents the likelihood
# that your classifier will assign a higher predicted probability to the positive observation.

# calculate a cross-validated AUC
from sklearn.model_selection import cross_val_score

print(
    f'Cross validation for AUC: {cross_val_score(logreg, X, y, cv=10, scoring="roc_auc").mean()}'
)
